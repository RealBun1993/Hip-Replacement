import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.utils import resample

# 固定随机种子
np.random.seed(42)

# 读取急性肾功能损伤模型数据
kidney_data = pd.read_csv('kidney_train_selected_features.csv', index_col='ID')

# 分离特征和目标变量（急性肾功能损伤模型）
X_kidney = kidney_data.drop(columns=['Cluster'])
y_kidney = kidney_data['Cluster']

# 获取类别特征和数值特征的列名
kidney_categorical_features = X_kidney.select_dtypes(include=['object', 'category']).columns.tolist()
kidney_numeric_features = X_kidney.select_dtypes(exclude=['object', 'category']).columns.tolist()

# 将类别特征转换为类别数据类型
for col in kidney_categorical_features:
    X_kidney[col] = X_kidney[col].astype('category')

# 拆分数据集（急性肾功能损伤模型）
X_kidney_train, X_kidney_test, y_kidney_train, y_kidney_test = train_test_split(X_kidney, y_kidney, test_size=0.3, random_state=42)

# 训练急性肾功能损伤模型
kidney_model = CatBoostClassifier(verbose=0, random_state=42, learning_rate=0.01, depth=5, iterations=300)
kidney_model.fit(X_kidney_train, y_kidney_train)

# 保存模型和特征名称（急性肾功能损伤模型）
joblib.dump(kidney_model, 'kidney_model.pkl')
joblib.dump(X_kidney_train.columns.tolist(), 'kidney_feature_names.pkl')  # 保存特征名称
joblib.dump(kidney_categorical_features, 'kidney_categorical_features.pkl')  # 保存类别特征名称

# 读取输血预测模型数据
blood_data = pd.read_csv('blood_train_selected_features.csv', index_col='ID')

# 分离特征和目标变量（输血预测模型）
X_blood = blood_data.drop(columns=['Cluster'])
y_blood = blood_data['Cluster']

# 获取类别特征和数值特征的列名
blood_categorical_features = X_blood.select_dtypes(include=['object', 'category']).columns.tolist()
blood_numeric_features = X_blood.select_dtypes(exclude=['object', 'category']).columns.tolist()

# 将类别特征转换为类别数据类型
for col in blood_categorical_features:
    X_blood[col] = X_blood[col].astype('category')

# 拆分数据集（输血预测模型）
X_blood_train, X_blood_test, y_blood_train, y_blood_test = train_test_split(X_blood, y_blood, test_size=0.3, random_state=42)

# 训练输血预测模型
blood_model = CatBoostClassifier(verbose=0, random_state=42)
blood_model.fit(X_blood_train, y_blood_train, cat_features=blood_categorical_features)

# 保存模型和特征名称（输血预测模型）
joblib.dump(blood_model, 'blood_model.pkl')
joblib.dump(X_blood_train.columns.tolist(), 'blood_feature_names.pkl')  # 保存特征名称
joblib.dump(blood_categorical_features, 'blood_categorical_features.pkl')  # 保存类别特征名称

# Flask 应用
app = Flask(__name__, static_folder='static')
CORS(app)

# 尝试加载急性肾功能损伤模型和特征名称
try:
    kidney_model = joblib.load('kidney_model.pkl')
    kidney_feature_names = joblib.load('kidney_feature_names.pkl')
    kidney_categorical_features = joblib.load('kidney_categorical_features.pkl')
    print(f"Loaded Kidney Model and Features")
except Exception as e:
    kidney_model = None
    kidney_feature_names = []
    kidney_categorical_features = []
    print(f"Error loading kidney model: {e}")

# 尝试加载输血预测模型和特征名称
try:
    blood_model = joblib.load('blood_model.pkl')
    blood_feature_names = joblib.load('blood_feature_names.pkl')
    blood_categorical_features = joblib.load('blood_categorical_features.pkl')
    print(f"Loaded Blood Model and Features")
except Exception as e:
    blood_model = None
    blood_feature_names = []
    blood_categorical_features = []
    print(f"Error loading blood model: {e}")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict_kidney', methods=['POST'])
def predict_kidney():
    if not kidney_model:
        return jsonify({'error': 'Kidney Model not loaded'}), 500

    data = request.json
    df = pd.DataFrame(data)
    df = df[kidney_feature_names]

    for col in kidney_categorical_features:
        df[col] = df[col].astype('category')

    prob_predictions = kidney_model.predict_proba(df)[:, 1]

    conf_intervals = []
    n_iterations = 1000
    noise_std = 0.4
    np.random.seed(42)
    for p in prob_predictions:
        bootstrapped_preds = []
        for i in range(n_iterations):
            boot_df = resample(df)
            noise = np.random.normal(0, noise_std, size=boot_df.shape)
            boot_df += noise
            boot_pred = kidney_model.predict_proba(boot_df)[:, 1].mean()
            bootstrapped_preds.append(boot_pred)
        
        lower = np.percentile(bootstrapped_preds, 2.5)
        upper = np.percentile(bootstrapped_preds, 97.5)
        conf_intervals.append((lower, upper))

    results = [{'probability': p, 'conf_interval': ci} for p, ci in zip(prob_predictions, conf_intervals)]
    return jsonify(results)

@app.route('/api/predict_blood', methods=['POST'])
def predict_blood():
    if not blood_model:
        return jsonify({'error': 'Blood Model not loaded'}), 500

    data = request.json
    df = pd.DataFrame(data)
    df = df[blood_feature_names]

    for col in blood_categorical_features:
        df[col] = df[col].astype('category')

    prob_predictions = blood_model.predict_proba(df)[:, 1]

    conf_intervals = []
    n_iterations = 1000
    noise_std = 0.4
    np.random.seed(42)
    for p in prob_predictions:
        bootstrapped_preds = []
        for i in range(n_iterations):
            boot_df = resample(df)
            noise = np.random.normal(0, noise_std, size=boot_df.shape)
            boot_df += noise
            boot_pred = blood_model.predict_proba(boot_df)[:, 1].mean()
            bootstrapped_preds.append(boot_pred)
        
        lower = np.percentile(bootstrapped_preds, 2.5)
        upper = np.percentile(bootstrapped_preds, 97.5)
        conf_intervals.append((lower, upper))

    results = [{'probability': p, 'conf_interval': ci} for p, ci in zip(prob_predictions, conf_intervals)]
    return jsonify(results)

@app.route('/api/feature_importance_kidney', methods=['GET'])
def feature_importance_kidney():
    if not kidney_model:
        return jsonify({'error': 'Kidney Model not loaded'}), 500

    importance = kidney_model.feature_importances_
    features = kidney_feature_names

    return jsonify(dict(zip(features, importance.tolist())))

@app.route('/api/feature_importance_blood', methods=['GET'])
def feature_importance_blood():
    if not blood_model:
        return jsonify({'error': 'Blood Model not loaded'}), 500

    importance = blood_model.feature_importances_
    features = blood_feature_names

    return jsonify(dict(zip(features, importance.tolist())))

@app.route('/api/model_summary_kidney', methods=['GET'])
def model_summary_kidney():
    if not kidney_model:
        return jsonify({'error': 'Kidney Model not loaded'}), 500

    summary = f"""
    Call:
    GradientBoostingClassifier with the following parameters:

    n_estimators: {kidney_model.get_params()['n_estimators']}
    max_depth: {kidney_model.get_params()['max_depth']}
    min_samples_split: {kidney_model.get_params()['min_samples_split']}
    min_samples_leaf: {kidney_model.get_params()['min_samples_leaf']}
    random_state: {kidney_model.get_params()['random_state']}

    Coefficients:
    {'Feature':<15}{'Importance':<15}
    {'-'*30}
    """
    for feature, importance in zip(kidney_feature_names, kidney_model.feature_importances_):
        summary += f"{feature:<15}{importance:<15.6f}\n"

    summary += f"\nAIC: Not applicable for tree-based models\n"
    summary += f"Model training score: {kidney_model.score(X_kidney_train, y_kidney_train):.4f}\n"
    summary += f"Model test score: {kidney_model.score(X_kidney_test, y_kidney_test):.4f}\n"
    summary += f"Number of Fisher Scoring iterations: Not applicable\n"

    return summary, 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/api/model_summary_blood', methods=['GET'])
def model_summary_blood():
    if not blood_model:
        return jsonify({'error': 'Blood Model not loaded'}), 500

    summary = f"""
    Call:
    CatBoostClassifier with the following parameters:

    iterations: {blood_model.get_param('iterations')}
    learning_rate: {blood_model.get_param('learning_rate')}
    depth: {blood_model.get_param('depth')}
    random_state: {blood_model.get_param('random_state')}

    Coefficients:
    {'Feature':<15}{'Importance':<15}
    {'-'*30}
    """
    for feature, importance in zip(blood_feature_names, blood_model.feature_importances_):
        summary += f"{feature:<15}{importance:<15.6f}\n"

    summary += f"\nAIC: Not applicable for tree-based models\n"
    summary += f"Model training score: {blood_model.score(X_blood_train, y_blood_train):.4f}\n"
    summary += f"Model test score: {blood_model.score(X_blood_test, y_blood_test):.4f}\n"
    summary += f"Number of Fisher Scoring iterations: Not applicable\n"

    return summary, 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/api/get_categorical_features_kidney', methods=['GET'])
def get_categorical_features_kidney():
    return jsonify(kidney_categorical_features)

@app.route('/api/get_categorical_features_blood', methods=['GET'])
def get_categorical_features_blood():
    return jsonify(blood_categorical_features)

@app.route('/api/get_unique_values_kidney', methods=['GET'])
def get_unique_values_kidney():
    feature = request.args.get('feature')
    unique_values = X_kidney_train[feature].unique().tolist()
    return jsonify(unique_values)

@app.route('/api/get_unique_values_blood', methods=['GET'])
def get_unique_values_blood():
    feature = request.args.get('feature')
    unique_values = X_blood_train[feature].unique().tolist()
    return jsonify(unique_values)

# if __name__ == '__main__':
    # app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
