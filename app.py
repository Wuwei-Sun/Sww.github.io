from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import QED

app = Flask(__name__)

# 加载模型
with open("D:\ml_web_app(1)\ml_web_app\model.pkl", "rb") as f:
    model = joblib.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # 渲染前端页面

@app.route("/predict", methods=["POST"])
# 定义预测函数
def predict():
    smiles = request.get_json()  # 获取前端发送的 JSON 数据
    """从SMILES字符串预测分子活性"""
    try:
        # 计算分子描述符（这里需要你实现描述符计算逻辑）
        def calculate_descriptors(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc = {
                    'SMILES': smiles,
                    'MolWt': Descriptors.MolWt(mol),
                    'NumAtoms': mol.GetNumAtoms(),
                    'NumBonds': mol.GetNumBonds(),
                    'ExactMolWt': Descriptors.ExactMolWt(mol),
                    'MolLogP': Descriptors.MolLogP(mol),
                    'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
                    'MaxEStateIndex':Descriptors.MaxEStateIndex(mol),
                    'MinEStateIndex':Descriptors.MinEStateIndex(mol),
                    'MinAbsEStateIndex':Descriptors.MinAbsEStateIndex(mol),
                    'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                    'FpDensityMorgan1': Descriptors.FpDensityMorgan1(mol),
                    'FpDensityMorgan2': Descriptors.FpDensityMorgan2(mol),
                    'FpDensityMorgan3': Descriptors.FpDensityMorgan3(mol),
                    'BCUT2D_MWHI': Descriptors.BCUT2D_MWHI(mol),
                    'BCUT2D_MWLOW': Descriptors.BCUT2D_MWLOW(mol),
                    'BCUT2D_CHGHI': Descriptors.BCUT2D_CHGHI(mol),
                    'BCUT2D_CHGLO': Descriptors.BCUT2D_CHGLO(mol),
                    'BCUT2D_LOGPHI': Descriptors.BCUT2D_LOGPHI(mol),
                    'BCUT2D_LOGPLOW': Descriptors.BCUT2D_LOGPLOW(mol),
                    'BCUT2D_MRLOW': Descriptors.BCUT2D_MRLOW(mol),
                    'BalabanJ': Descriptors.BalabanJ(mol),
                    'BertzCT': Descriptors.BertzCT(mol),
                    'Chi0': Descriptors.Chi0(mol),
                    'Chi0n': Descriptors.Chi0n(mol),
                    'Chi0v': Descriptors.Chi0v(mol),
                    'Chi1': Descriptors.Chi1(mol),
                    'Chi1n': Descriptors.Chi1n(mol),
                    'Chi1v': Descriptors.Chi1v(mol),
                    'Chi2n': Descriptors.Chi2n(mol),
                    'Chi2v': Descriptors.Chi2v(mol),
                    'Chi3n': Descriptors.Chi3n(mol),
                    'Chi3v': Descriptors.Chi3v(mol),
                    'Chi4n': Descriptors.Chi4n(mol),
                    'Chi4v': Descriptors.Chi4v(mol),
                    'Kappa1': Descriptors.Kappa1(mol),
                    'Kappa2': GraphDescriptors.Kappa2(mol),
                    'Kappa3': GraphDescriptors.Kappa3(mol),
                    'LabuteASA': Descriptors.LabuteASA(mol),
                    'PEOE_VSA6': Descriptors.PEOE_VSA6(mol),
                    'PEOE_VSA7': Descriptors.PEOE_VSA7(mol),
                    'PEOE_VSA8': Descriptors.PEOE_VSA8(mol),
                    'SMR_VSA10': Descriptors.SMR_VSA10(mol),
                    'SMR_VSA6': Descriptors.SMR_VSA6(mol),
                    'SMR_VSA7': Descriptors.SMR_VSA7(mol),
                    'SlogP_VSA1': Descriptors.SlogP_VSA1(mol),
                    'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),
                    'SlogP_VSA3': Descriptors.SlogP_VSA3(mol),
                    'SlogP_VSA6': Descriptors.SlogP_VSA6(mol),
                    'EState_VSA2': Descriptors.EState_VSA2(mol),
                    'EState_VSA4': Descriptors.EState_VSA4(mol),
                    'EState_VSA6': Descriptors.EState_VSA6(mol),
                    'VSA_EState2': Descriptors.VSA_EState2(mol),
                    'VSA_EState3': Descriptors.VSA_EState3(mol),
                    'VSA_EState4': Descriptors.VSA_EState4(mol),
                    'VSA_EState5': Descriptors.VSA_EState5(mol),
                    'VSA_EState6': Descriptors.VSA_EState6(mol),
                    'VSA_EState7': Descriptors.VSA_EState7(mol),
                    'RingCount': Descriptors.RingCount(mol),
                     }
                return desc

        # 定义特征顺序（需与模型训练时使用的特征顺序一致）
        FEATURE_ORDER = ['MolWt', 'NumAtoms', 'NumBonds', 'ExactMolWt', 'MolLogP', 'NumHeavyAtoms', 'MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA6', 'SMR_VSA7', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA6', 'EState_VSA2', 'EState_VSA4', 'EState_VSA6', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'RingCount']
                
        #调用函数计算分子描述符
        desc_dict = calculate_descriptors(smiles)

         # 按特征顺序提取值，转换为二维数组（model.predict要求输入为 (样本数, 特征数)）
        desc_values = [desc_dict[feat] for feat in FEATURE_ORDER]  
        desc_df = pd.DataFrame([desc_values], columns=FEATURE_ORDER)  
        
        # 4. 模型预测 + 结果解析（修复三元表达式，补充else）
        pred_label = model.predict(desc_df)[0]  # 无错误

        
        # 完整三元表达式：条件成立返回Active，否则返回Inactive
        pred_result = "Active（活性）" if pred_label == 1 else "Inactive（非活性）"
        
        # 返回结果
        return jsonify({
            'prediction': pred_result,
            'smiles': smiles,
            'pred_label': int(pred_label)
        })
    
    # 捕获所有可能的异常（如无效SMILES、模型未加载等）
    except Exception as e:
        return jsonify({'error': f'处理SMILES时出错: {str(e)}'}), 400
    
    


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)