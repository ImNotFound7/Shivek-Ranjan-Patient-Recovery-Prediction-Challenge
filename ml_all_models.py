import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
# import warnings
# warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

#=============================================================================
# SMART DATA CONVERSION
#=============================================================================

def convert_binary_columns(df):
    df_converted = df.copy()
    conversions_made = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            unique_vals_lower = [str(v).lower().strip() for v in unique_vals]

            if len(unique_vals) <= 2:
                binary_mappings = {
                    ('yes', 'no'): {'yes': 1, 'no': 0},
                    ('y', 'n'): {'y': 1, 'n': 0},
                    ('true', 'false'): {'true': 1, 'false': 0},
                    ('t', 'f'): {'t': 1, 'f': 0},
                    ('1', '0'): {'1': 1, '0': 0},
                    ('1.0', '0.0'): {'1.0': 1, '0.0': 0},
                    ('positive', 'negative'): {'positive': 1, 'negative': 0},
                    ('pos', 'neg'): {'pos': 1, 'neg': 0},
                    ('present', 'absent'): {'present': 1, 'absent': 0},
                    ('active', 'inactive'): {'active': 1, 'inactive': 0},
                }

                unique_set = set(unique_vals_lower)
                for key_pair, mapping in binary_mappings.items():
                    if unique_set.issubset(set(key_pair)):
                        case_insensitive_map = {}
                        for orig_val in unique_vals:
                            orig_lower = str(orig_val).lower().strip()
                            if orig_lower in mapping:
                                case_insensitive_map[orig_val] = mapping[orig_lower]

                        df_converted[col] = df[col].map(case_insensitive_map)
                        conversions_made[col] = f"Converted to binary: {dict(case_insensitive_map)}"
                        break

    return df_converted, conversions_made


def smart_data_preparation(df, target_col=None):
    df_clean = df.copy()

    if target_col and target_col in df.columns:
        y = df[target_col]
        df_clean = df.drop(target_col, axis=1)
    else:
        y = None

    if 'Id' in df_clean.columns:
        df_clean = df_clean.drop('Id', axis=1)

    print(f"\nInitial features: {list(df_clean.columns)}")
    print(f"Initial shape: {df_clean.shape}")

    df_clean, conversions = convert_binary_columns(df_clean)

    if conversions:
        for col, conversion_info in conversions.items():
            print(f"  ✓ {col}: {conversion_info}")
    else:
        print("  No binary text columns found")

    numeric_features = []
    categorical_features = []

    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
        else:
            converted = pd.to_numeric(df_clean[col], errors='coerce')
            non_null_ratio = converted.notna().sum() / len(df_clean)

            if non_null_ratio >= 0.9:
                df_clean[col] = converted
                numeric_features.append(col)
                print(f"  ✓ {col}: Converted to numeric ({non_null_ratio*100:.1f}% success)")
            elif non_null_ratio > 0:
                categorical_features.append(col)
                print(f"  → {col}: Kept as categorical (only {non_null_ratio*100:.1f}% numeric)")
            else:
                unique_count = df_clean[col].nunique()
                if unique_count > 0 and unique_count < 50:
                    categorical_features.append(col)
                    print(f"  → {col}: Kept as categorical ({unique_count} unique values)")
                else:
                    print(f"  ✗ {col}: Cannot use (all missing or too many categories)")

    for col in numeric_features:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"  Filled {missing_count} missing in '{col}' with median: {median_val:.2f}")

    for col in categorical_features:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"  Filled {missing_count} missing in '{col}'")

    print(f"\nFinal feature summary:")
    print(f"  Numeric: ({len(numeric_features)}): {numeric_features}")
    print(f"  Categorical: ({len(categorical_features)}): {categorical_features}")
    print(f"  Shape: {df_clean.shape}")

    if y is not None:
        return df_clean, y, numeric_features, categorical_features
    else:
        return df_clean, numeric_features, categorical_features


def create_engineered_features(df, numeric_features):
    df_new = df.copy()
    valid_numeric = [f for f in numeric_features if f in df.columns]

    if len(valid_numeric) < 2:
        return df_new

    print("\nCreating engineered features:")

    interactions = 0
    for i, col1 in enumerate(valid_numeric):
        for col2 in valid_numeric[i+1:]:
            try:
                df_new[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                interactions += 1
            except:
                pass

    squared = 0
    for col in valid_numeric:
        try:
            df_new[f'{col}_squared'] = df[col] ** 2
            squared += 1
        except:
            pass

    ratios = 0
    for i, col1 in enumerate(valid_numeric):
        for col2 in valid_numeric[i+1:]:
            try:
                df_new[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 0.01)
                ratios += 1
            except:
                pass

    print(f"  Interactions: {interactions}, Squared: {squared}, Ratios: {ratios}")
    print(f"  Total features: {df_new.shape[1]}")

    return df_new

#=============================================================================
# PREPROCESSING AND MODELS
#=============================================================================

def create_preprocessing_pipeline(numeric_features, categorical_features):
    transformers = []

    if len(numeric_features) > 0:
        transformers.append(('num', StandardScaler(), numeric_features))

    if len(categorical_features) > 0:
        transformers.append(('cat', 
                           OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                           categorical_features))

    return ColumnTransformer(transformers=transformers, remainder='drop')


def get_models():

    return {
        'Linear Regression (degree=1)': {
            'model': LinearRegression(),
            'params': {},
            'use_poly': False,
            'description': 'Standard OLS - baseline'
        },
        'Polynomial LR (degree=2)': {
            'model': LinearRegression(),
            'params': {},
            'use_poly': True,
            'poly_degree': 2,
            'description': 'Adds x², xy interactions'
        },
        'Polynomial LR (degree=3)': {
            'model': LinearRegression(),
            'params': {},
            'use_poly': True,
            'poly_degree': 3,
            'description': 'Adds x³, x²y interactions'
        },
        'Ridge': {
            'model': Ridge(random_state=RANDOM_STATE),
            'params': {'alpha': [0.01, 0.1, 1, 10, 100]},
            'use_poly': False
        },
        'Lasso': {
            'model': Lasso(random_state=RANDOM_STATE, max_iter=5000),
            'params': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
            'use_poly': False
        },
        'ElasticNet': {
            'model': ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
            'params': {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.3, 0.5, 0.7, 0.9]},
            'use_poly': False
        },
        'BayesianRidge': {
            'model': BayesianRidge(),
            'params': {},
            'use_poly': False
        },
        'DecisionTree': {
            'model': DecisionTreeRegressor(random_state=RANDOM_STATE),
            'params': {'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 5, 10]},
            'use_poly': False
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
            'params': {
                'n_estimators': [300, 500, 700],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'randomized': True,
            'use_poly': False
        },
        # 'LightGBM': {
        #     'model': LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        #     'params': {
        #         'n_estimators': [300, 500, 700],
        #         'max_depth': [5, 7, 10],
        #         'learning_rate': [0.01, 0.05, 0.1],
        #         'num_leaves': [31, 63, 127]
        #     },
        #     'randomized': True,
        #     'use_poly': False
        # },
        'AdaBoost': {
            'model': AdaBoostRegressor(random_state=RANDOM_STATE),
            'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]},
            'use_poly': False
        }
    }


def train_model(X, y, name, config, numeric_features, categorical_features, cv=5):

    # Check if polynomial features should be added
    use_poly = config.get('use_poly', False)
    poly_degree = config.get('poly_degree', 2)

    # Create pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

    if use_poly:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])

    params = config.get('params', {})
    use_randomized = config.get('randomized', False)

    if params:
        param_grid = {f'model__{k}': v for k, v in params.items()}

        if use_randomized:
            search = RandomizedSearchCV(pipeline, param_grid, n_iter=20, cv=cv,
                                       scoring='neg_root_mean_squared_error',
                                       n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
        else:
            search = GridSearchCV(pipeline, param_grid, cv=cv,
                                 scoring='neg_root_mean_squared_error',
                                 n_jobs=-1, verbose=0)

        search.fit(X, y)
        return {
            'name': name,
            'model': search.best_estimator_,
            'cv_rmse': -search.best_score_,
            'params': search.best_params_
        }
    else:
        scores = cross_val_score(pipeline, X, y, cv=cv,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)
        pipeline.fit(X, y)
        return {
            'name': name,
            'model': pipeline,
            'cv_rmse': -scores.mean(),
            'params': {}
        }


def train_all_models(X, y, numeric_features, categorical_features, cv=5):
    models = get_models()
    results = []

    print("\n" + "=" * 80)
    print("TRAINING MODELS (3 Linear Regression variants + 8 others)")
    print("=" * 80)

    for name, config in models.items():
        desc = config.get('description', '')
        desc_str = f" ({desc})" if desc else ""
        print(f"\nTraining {name}{desc_str}...", end='', flush=True)
        try:
            result = train_model(X, y, name, config, numeric_features, categorical_features, cv)
            results.append(result)
            print(f"  CV RMSE: {result['cv_rmse']:.4f}")
        except Exception as e:
            print(f"  Error: {str(e)}")

    return results


def create_ensemble(X, y, results, numeric_features, categorical_features, cv=5):
    if len(results) < 3:
        return None

    print("\n" + "=" * 80)
    print("CREATING STACKING ENSEMBLE")
    print("=" * 80)

    sorted_results = sorted(results, key=lambda x: x['cv_rmse'])[:3]
    print(f"\nTop 3 models:")
    for r in sorted_results:
        print(f"  - {r['name']}: {r['cv_rmse']:.4f}")

    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    estimators = [(r['name'], r['model'].named_steps['model'] if 'model' in r['model'].named_steps else r['model'].named_steps['poly']) for r in sorted_results]

    stacking = Pipeline([
        ('preprocessor', preprocessor),
        ('stacker', StackingRegressor(estimators=estimators,
                                     final_estimator=Ridge(alpha=1.0),
                                     cv=cv, n_jobs=-1))
    ])

    scores = cross_val_score(stacking, X, y, cv=cv,
                            scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_rmse = -scores.mean()

    print(f"\nStacking Ensemble CV RMSE: {cv_rmse:.4f}")

    stacking.fit(X, y)

    return {'name': 'Stacking Ensemble', 'model': stacking, 'cv_rmse': cv_rmse, 'params': {}}

#=============================================================================
# MODIFIED MAIN PIPELINE - GENERATES PREDICTIONS FOR ALL MODELS
#=============================================================================

def run_pipeline_all_models(train_path, test_path=None, output_folder='predictions_all_models',
                            engineer_features=True, use_ensemble=True):
    import os

    # Create output folder if it doesn't exist
    if test_path and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\nCreated output folder: {output_folder}")

    print("=" * 80)
    print("SMART PIPELINE - Generating Predictions for ALL Models")
    print("=" * 80)

    print("\n[Step 1/5] Loading data...")
    train_df = pd.read_csv(train_path)
    X, y, numeric_features, categorical_features = smart_data_preparation(
        train_df, target_col='Recovery Index'
    )

    print(f"\nTarget: Mean={y.mean():.2f}, Std={y.std():.2f}, Range=[{y.min():.2f}, {y.max():.2f}]")

    if engineer_features and len(numeric_features) >= 2:
        print("\n[Step 2/5] Creating engineered features...")
        X_cols_before = list(X.columns)
        X = create_engineered_features(X, numeric_features)
        new_features = [c for c in X.columns if c not in X_cols_before]
        numeric_features = numeric_features + new_features
    else:
        print("\n[Step 2/5] Skipping feature engineering")

    print("\n[Step 3/5] Training models...")
    results = train_all_models(X, y, numeric_features, categorical_features, cv=5)

    if use_ensemble and len(results) >= 3:
        print("\n[Step 4/5] Creating ensemble...")
        ensemble = create_ensemble(X, y, results, numeric_features, categorical_features, cv=5)
        if ensemble:
            results.append(ensemble)
    else:
        print("\n[Step 4/5] Skipping ensemble")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    results_df = pd.DataFrame([
        {'Model': r['name'], 'CV RMSE': r['cv_rmse']}
        for r in results
    ]).sort_values('CV RMSE')

    print("\n" + results_df.to_string(index=False))

    best = min(results, key=lambda x: x['cv_rmse'])
    print(f"\nBest Model: {best['name']}")
    print(f"Best CV RMSE: {best['cv_rmse']:.4f}")

    # Compare Linear Regression variants
    linear_results = [r for r in results if 'Linear' in r['name'] or 'Polynomial LR' in r['name']]
    if linear_results:
        print("\n📊 Linear Regression Comparison:")
        for r in sorted(linear_results, key=lambda x: x['cv_rmse']):
            print(f"   {r['name']}: {r['cv_rmse']:.4f}")

    if test_path:
        print("\n[Step 5/5] Generating predictions for ALL models...")
        test_df = pd.read_csv(test_path)
        test_ids = test_df['Id'].copy() if 'Id' in test_df.columns else None

        X_test, test_num, test_cat = smart_data_preparation(test_df)

        if engineer_features and len(numeric_features) >= 2:
            X_test = create_engineered_features(X_test, test_num)

        print("\n" + "=" * 80)
        print("GENERATING SEPARATE PREDICTIONS FOR EACH MODEL")
        print("=" * 80)

        all_predictions = {}

        for i, result in enumerate(results, 1):
            model_name = result['name']
            model = result['model']

            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            filename = f"{output_folder}/{safe_name}.csv"

            print(f"\n{i}/{len(results)} Generating predictions for: {model_name}...", end='', flush=True)

            try:
                predictions = model.predict(X_test)

                all_predictions[model_name] = predictions

                submission = pd.DataFrame({
                    'Id': test_ids if test_ids is not None else range(len(predictions)),
                    'Recovery Index': predictions
                })
                submission.to_csv(filename, index=False)

                print(f" ✓")
                print(f"      Saved: {filename}")
                print(f"      Mean={predictions.mean():.2f}, Std={predictions.std():.2f}, Range=[{predictions.min():.2f}, {predictions.max():.2f}]")

            except Exception as e:
                print(f" ✗ Error: {str(e)}")

        print("\n" + "=" * 80)
        print("BONUS: Creating Averaged Prediction from Top 3 Models")
        print("=" * 80)

        top_3 = sorted(results, key=lambda x: x['cv_rmse'])[:3]
        print("\nAveraging predictions from:")
        for i, m in enumerate(top_3, 1):
            print(f"  {i}. {m['name']} (CV RMSE: {m['cv_rmse']:.4f})")

        top_3_preds = [all_predictions[m['name']] for m in top_3 if m['name'] in all_predictions]

        if len(top_3_preds) == 3:
            averaged_preds = np.mean(top_3_preds, axis=0)

            avg_submission = pd.DataFrame({
                'Id': test_ids if test_ids is not None else range(len(averaged_preds)),
                'Recovery Index': averaged_preds
            })
            avg_filename = f"{output_folder}/TOP3_AVERAGED.csv"
            avg_submission.to_csv(avg_filename, index=False)

            print(f"\n✓ Saved averaged predictions: {avg_filename}")
            print(f"  Mean={averaged_preds.mean():.2f}, Std={averaged_preds.std():.2f}")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nGenerated {len(all_predictions)} prediction files")
        print(f"All files saved in: {output_folder}/")
        print(f"\nFiles created:")

        for i, (model_name, preds) in enumerate(all_predictions.items(), 1):
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            print(f"   {i:2d}. {safe_name}.csv - {model_name}")

        print(f"   {len(all_predictions)+1:2d}. TOP3_AVERAGED.csv - Average of top 3 models")

        print("\nRECOMMENDATION:")
        print(f"   - Start by submitting: {output_folder}/TOP3_AVERAGED.csv")
        print(f"   - Then try: {top_3[0]['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.csv")
        print(f"   - Compare scores and pick the best!")

    else:
        print("\n[Step 5/5] No test file provided")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)

    return results_df, results


if __name__ == "__main__":
    TRAIN_PATH = 'train.csv'
    TEST_PATH = 'test.csv'
    OUTPUT_FOLDER = 'predictions_all_models'  # Folder to save all predictions

    results_df, all_results = run_pipeline_all_models(
        TRAIN_PATH, TEST_PATH, OUTPUT_FOLDER,
        engineer_features=True, use_ensemble=True
    )