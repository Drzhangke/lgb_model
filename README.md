# 胆囊癌与黄色肉芽肿性胆囊炎鉴别诊断系统

这是一个基于LightGBM机器学习模型的医疗辅助诊断系统，用于鉴别诊断胆囊癌(GBC)和黄色肉芽肿性胆囊炎(XGC)。

## 功能特点

- 基于12个临床和影像学参数进行预测
- 使用LightGBM机器学习算法
- 提供友好的Streamlit Web界面
- 完全免费部署方案

## 安装和运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型文件

系统支持多种模型格式，按优先级自动选择：

1. **R语言模型** (`lgb_model.rds`) - 推荐使用，保持原始训练效果
2. **Python模型** (`lgb_model.pkl`) - 转换后的Python格式模型
3. **示例模型** - 当以上模型都不可用时自动创建

### 3. 安装R环境（可选，用于rpy2）

如果要使用R语言模型，需要安装R环境和相关包：

```bash
# 安装R (Windows)
# 从 https://cran.r-project.org/bin/windows/base/ 下载并安装R

# 在R中安装必要包
install.packages(c("lightgbm", "dplyr", "plumber"))
```

### 4. 运行应用

```bash
streamlit run app.py
```

## 部署到Streamlit Community Cloud

1. 将代码推送到GitHub仓库
2. 访问 [Streamlit Community Cloud](https://streamlit.io/cloud)
3. 使用GitHub账号登录
4. 点击"New app"按钮
5. 选择你的GitHub仓库和`app.py`文件
6. 点击"Deploy"完成部署

## 参数说明

系统需要输入以下12个参数：

### 临床特征
- 性别(Sex)
- 胆囊结石(Gallstone)
- 其他疾病(Other)

### 影像特征
- 超声回声(Ultrasound Echo)
- 胆管扩张(Biliary Duct Dilation)
- 胆囊形态(Gallbladder Morphology)
- 胆囊壁结节(Intramural Nodules)
- 腔内肿块(Intraluminal Tumor)
- 粘膜线(Mucosal Line)
- 肿大淋巴结(Enlarged Lymph Nodes)

### 实验室检查
- 纤维蛋白原(Fibrinogen, g/L)
- 间接胆红素(Indirect Bilirubin, µmol/L)

## 使用说明

1. 在左侧表单中输入患者的各项检查参数
2. 系统会自动实时进行诊断并显示结果
3. 修改任何参数都会立即更新诊断结果
4. 本系统仅作为辅助诊断工具，不能替代医生的专业判断

## 技术架构

- **前端**: Streamlit
- **后端**: Python + LightGBM + rpy2 (可选)
- **模型支持**:
  - R语言LightGBM模型 (.rds格式)
  - Python LightGBM模型 (.pkl格式)
  - 自动降级到示例模型
- **部署**: Streamlit Community Cloud (完全免费)

### 模型加载优先级

1. **R模型** (.rds) - 使用rpy2直接调用，保持最佳性能
2. **Python模型** (.pkl) - 序列化的Python模型
3. **示例模型** - 基于模拟数据的LightGBM模型

## R模型转换方案

由于R语言和Python环境的兼容性问题，系统提供了多种R模型转换方案：

### 方案1：rpy2直接调用（推荐）
```bash
# 安装R环境和rpy2
pip install rpy2

# 确保R_HOME环境变量正确设置
# Windows: set R_HOME=C:\Program Files\R\R-4.x.x
```

### 方案2：参数导出转换
```bash
# 在R环境中运行导出脚本
Rscript export_r_model.R

# 这将生成 lgb_model_params.json 文件
# Python应用会自动读取此文件进行预测
```

### 方案3：自动降级
如果以上方案都不可用，系统会自动使用Python LightGBM示例模型，确保应用正常运行。

## 使用指南

### 1. 优先级说明
系统按以下优先级自动选择模型：
1. **R模型 (.rds)** + rpy2 - 最准确，保持原始训练效果
2. **R模型参数 (JSON)** - 近似准确，轻量级
3. **Python示例模型** - 保证可用性

### 2. 最佳实践
- **科研环境**：建议安装R + rpy2，使用原始R模型
- **生产环境**：使用参数导出方案，避免R依赖
- **演示环境**：使用示例模型，快速部署

## 注意事项

- 本系统仅供医学研究和临床参考使用
- 诊断结果不能替代医生的专业判断
- 示例模型基于模拟数据，仅用于功能演示
- 如需使用真实模型，请训练并保存LightGBM模型