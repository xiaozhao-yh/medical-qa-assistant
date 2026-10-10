# 医疗知识问答助手（增强版）

项目简介
- 基于 RAG（检索增强生成）的医疗知识问答助手
- 支持上传病历或文本文件（txt / md / pdf）
- 支持多轮问答、短/长回答、中文/English
- 支持登录认证
- 调用阿里云百炼 Qwen（DashScope）进行回答
- 本系统仅供学习与参考，不构成医疗建议

运行环境
- Python 3.8+
- PyCharm

安装依赖
1. 创建虚拟环境
2. 执行：
   pip install -r requirements.txt

初始化知识库
1. 进入项目根目录
2. 执行：
   python build_index.py

设置环境变量
- Linux/macOS:
  export DASHSCOPE_API_KEY="你的百炼API Key"
  export DASHSCOPE_MODEL="qwen-plus"
- Windows PowerShell:
  $env:DASHSCOPE_API_KEY="你的百炼API Key"
  $env:DASHSCOPE_MODEL="qwen-plus"

启动应用
python app.py

登录方式
- 用户名: testuser
- 密码: password

注意事项
- 本系统仅供学习参考，不构成医疗建议
- 医疗相关内容请以医生和专业医疗机构意见为准
