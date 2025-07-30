# 简化版本 - 直接测试 SkillExplainer 加载
import os
import sys

print("=" * 60)
print("🚀 SIMPLE TEST VERSION")
print("=" * 60)

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# 检查环境变量
openai_key = os.getenv('OPENAI_API_KEY')
print(f"🔍 OPENAI_API_KEY exists: {openai_key is not None}")
if openai_key:
    print(f"🔍 Key length: {len(openai_key)}")
    print(f"🔍 Key starts with: {openai_key[:15]}")

# 强制加载 SkillExplainer
skill_explainer = None
try:
    print("🔄 Attempting to load SkillExplainer...")
    
    # 方法1: 从 models 目录
    try:
        from skill_explainer import SkillExplainer
        skill_explainer = SkillExplainer()
        print("✅ Loaded from models directory")
    except ImportError as e1:
        print(f"❌ Failed from models: {e1}")
        
        # 方法2: 从根目录
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from skill_explainer import SkillExplainer
            skill_explainer = SkillExplainer()
            print("✅ Loaded from root directory")
        except ImportError as e2:
            print(f"❌ Failed from root: {e2}")
            
            # 方法3: 使用完整路径
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "skill_explainer", 
                    os.path.join(os.path.dirname(__file__), "skill_explainer.py")
                )
                if spec and spec.loader:
                    skill_explainer_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(skill_explainer_module)
                    skill_explainer = skill_explainer_module.SkillExplainer()
                    print("✅ Loaded using importlib")
                else:
                    raise ImportError("Could not create spec")
            except Exception as e3:
                print(f"❌ Failed with importlib: {e3}")
                raise ImportError("All import methods failed")

    if skill_explainer:
        print(f"🔍 SkillExplainer available: {skill_explainer.is_available()}")
        print(f"🔍 SkillExplainer type: {type(skill_explainer).__name__}")
        
        # 测试功能
        try:
            test_result = skill_explainer.explain_skills(['Python'], 'Developer')
            print(f"🔍 Test result success: {test_result.get('success', False)}")
        except Exception as test_e:
            print(f"❌ Test failed: {test_e}")
    else:
        print("❌ SkillExplainer is None")

except Exception as e:
    print(f"❌ Critical error loading SkillExplainer: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)

# 现在创建Flask应用
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'success': True,
        'message': 'Simple Test API',
        'features': {
            'ai_skill_explanations': skill_explainer.is_available() if skill_explainer else False
        },
        'debug_info': {
            'skill_explainer_exists': skill_explainer is not None,
            'skill_explainer_type': type(skill_explainer).__name__ if skill_explainer else None,
            'openai_key_exists': openai_key is not None,
            'openai_key_length': len(openai_key) if openai_key else 0
        }
    })

@app.route('/test-skill-explainer')
def test_skill_explainer():
    if not skill_explainer:
        return jsonify({
            'success': False,
            'error': 'SkillExplainer not loaded'
        })
    
    try:
        result = skill_explainer.explain_skills(['Python', 'JavaScript'], 'Developer')
        return jsonify({
            'success': True,
            'available': skill_explainer.is_available(),
            'test_result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'available': skill_explainer.is_available() if skill_explainer else False
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
