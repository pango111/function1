#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Job Classification and Skill Extraction Web Application: Flask Backend
Optimized version with batch processing support
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import sys
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import io
import zipfile
import csv
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from collections import Counter

# Add models directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS Configuration
if os.getenv('ENVIRONMENT') == 'production':
    CORS(app, origins=[
        'https://storage.googleapis.com',
        'https://storage.cloud.google.com',
    ])
else:
    # 开发环境 - 允许你的Vercel网站访问
    CORS(app, origins=[
        'http://localhost:3000',
        'https://demo-roan-theta.vercel.app',  # 你的Vercel网站
        'https://*.vercel.app'  # 允许所有vercel.app域名
    ])

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

if os.getenv('ENVIRONMENT') != 'production':
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
classifier = None
model_loaded = False
skill_extractor = None
extractor_mode = None
skill_explainer = None
batch_results = {}  # 存储批量处理结果
batch_lock = threading.Lock()

# Job category mapping
JOB_CATEGORIES = {
    0: {
        "name": "CyberSecurity Consultant",
        "description": "Cybersecurity consulting and advisory roles",
        "color": "#E91E63",
    },
    1: {
        "name": "CyberSecurity Analyst", 
        "description": "Security analysis and monitoring positions",
        "color": "#2196F3",
    },
    2: {
        "name": "CyberSecurity Architect",
        "description": "Security architecture and design roles",
        "color": "#9C27B0",
    },
    3: {
        "name": "CyberSecurity Operations",
        "description": "Security operations and incident response",
        "color": "#FF5722",
    },
    4: {
        "name": "Information Security",
        "description": "Information security management and governance",
        "color": "#607D8B",
    },
    5: {
        "name": "CyberSecurity Testers",
        "description": "Penetration testing and security assessment",
        "color": "#FF9800",
    }
}

def load_model():
    """Load model and skill extractors with correct import paths"""
    global classifier, model_loaded, skill_extractor, extractor_mode, skill_explainer

    try:
        # Try to load trained model
        model_path = 'job_classifier_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
            model_loaded = True
            logger.info("✅ Trained JobClassifier model loaded successfully")
        else:
            logger.warning("⚠️  Model file not found. Using demo mode.")
            model_loaded = False

        # Try to load skill extractors from models directory
        skill_extractor = None
        try:
            # Try HybridSkillExtractor first
            try:
                from model import HybridSkillExtractor
                skill_extractor = HybridSkillExtractor()
                
                if hasattr(skill_extractor, 'get_extraction_info'):
                    extraction_info = skill_extractor.get_extraction_info()
                    extractor_mode = extraction_info.get('extraction_mode', 'hybrid')
                    
                    if extraction_info.get('textrank_available', False):
                        logger.info("✅ HybridSkillExtractor loaded with spaCy + TextRank")
                    else:
                        logger.info("✅ HybridSkillExtractor loaded (rule-based only)")
                else:
                    extractor_mode = 'hybrid'
                    logger.info("✅ HybridSkillExtractor loaded")
                    
            except Exception as e:
                logger.warning(f"⚠️  HybridSkillExtractor failed: {e}")
                
                # Fallback to BasicSkillExtractor from models/demo.py
                try:
                    from demo import BasicSkillExtractor
                    skill_extractor = BasicSkillExtractor()
                    extractor_mode = 'basic'
                    logger.info("✅ BasicSkillExtractor loaded from models/demo.py")
                except Exception as e2:
                    logger.warning(f"⚠️  BasicSkillExtractor from models failed: {e2}")
                    
                    # Try from models/classifier.py
                    try:
                        from classifier import BasicSkillExtractor
                        skill_extractor = BasicSkillExtractor()
                        extractor_mode = 'basic'
                        logger.info("✅ BasicSkillExtractor loaded from models/classifier.py")
                    except Exception as e3:
                        logger.error(f"❌ All skill extractors failed: {e3}")
                        skill_extractor = create_minimal_extractor()
                        extractor_mode = 'minimal'

        except Exception as e:
            logger.error(f"❌ Failed to load any skill extractor: {e}")
            skill_extractor = create_minimal_extractor()
            extractor_mode = 'minimal'

        # Try to load AI skill explainer
        try:
            # Try from models directory first
            try:
                from skill_explainer import SkillExplainer
                skill_explainer = SkillExplainer()
            except ImportError:
                # Try from root directory
                sys.path.append(os.path.dirname(__file__))
                from skill_explainer import SkillExplainer
                skill_explainer = SkillExplainer()
            
            if skill_explainer.is_available():
                logger.info("✅ AI skill explainer loaded and ready")
            else:
                logger.warning("⚠️  AI skill explainer loaded but OpenAI API key not available")
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to load skill explainer: {e}")
            skill_explainer = create_minimal_explainer()

        return True

    except Exception as e:
        logger.error(f"❌ Critical error in load_model: {e}")
        # Ensure minimal functionality
        if skill_extractor is None:
            skill_extractor = create_minimal_extractor()
            extractor_mode = 'minimal'
        if skill_explainer is None:
            skill_explainer = create_minimal_explainer()
        return False

def create_minimal_extractor():
    """Create a minimal skill extractor if imports fail"""
    class MinimalSkillExtractor:
        def extract_all_skills(self, text):
            if not text:
                return []
            text_lower = text.lower()
            basic_skills = [
                'python', 'java', 'javascript', 'cybersecurity', 'aws', 'azure',
                'docker', 'kubernetes', 'mysql', 'react', 'angular', 'git',
                'penetration testing', 'incident response', 'siem', 'firewall',
                'burp suite', 'nmap', 'metasploit', 'linux', 'windows'
            ]
            found_skills = []
            for skill in basic_skills:
                if skill in text_lower:
                    found_skills.append(skill)
            return found_skills
        
        def extract_skills(self, text):
            skills = self.extract_all_skills(text)
            return {
                "rule_based_skills": skills,
                "textrank_phrases": [],
                "combined_skills": skills
            }
    
    logger.info("✅ MinimalSkillExtractor created as fallback")
    return MinimalSkillExtractor()

def create_minimal_explainer():
    """Create a minimal skill explainer if imports fail"""
    class MinimalSkillExplainer:
        def is_available(self):
            return False
        
        def explain_skills(self, skills, job_category=""):
            # Basic explanations for common skills
            basic_explanations = {
                'python': 'Python is a popular programming language used for automation, scripting, and application development.',
                'java': 'Java is an object-oriented programming language. Study Java fundamentals, OOP concepts, and frameworks.',
                'javascript': 'JavaScript is essential for web development. Learn JS fundamentals, DOM manipulation, and modern frameworks.',
                'aws': 'Amazon Web Services (AWS) is a cloud computing platform. Learn AWS services and cloud concepts.',
                'docker': 'Docker is a containerization platform. Learn how to create, manage, and deploy containers.',
                'cybersecurity': 'Cybersecurity protects systems from digital threats. Study security principles and defense strategies.',
                'penetration testing': 'Penetration testing involves ethical hacking to find security vulnerabilities.',
                'siem': 'Security Information and Event Management (SIEM) systems monitor security events.',
                'linux': 'Linux is an operating system crucial for many IT roles. Learn command line and system administration.'
            }
            
            explanations = {}
            for skill in skills:
                skill_lower = skill.lower()
                if skill_lower in basic_explanations:
                    explanations[skill] = basic_explanations[skill_lower]
                else:
                    explanations[skill] = f"{skill} is a technology/skill required for this position."
            
            return {
                'success': False,
                'error': 'AI skill explainer not available - using fallback explanations',
                'skill_explanations': explanations,
                'skills_count': len(skills),
                'generation_method': 'fallback'
            }
    
    return MinimalSkillExplainer()

def classify_job_category(text):
    """Rule-based job classification fallback"""
    text = text.lower()

    category_keywords = {
        1: ['analyst', 'monitoring', 'detection', 'incident analyst', 'blue team', 'soc analyst'],
        2: ['architect', 'architecture', 'design security', 'security by design', 'solution architect'],
        3: ['devops', 'operations', 'infrastructure', 'system admin', 'cloud ops', 'devsecops'],
        4: ['governance', 'compliance', 'risk management', 'grc', 'ciso', 'policy', 'information security'],
        5: ['penetration testing', 'pentester', 'ethical hacker', 'red team', 'vulnerability', 'pen test'],
        0: ['consultant', 'advisory', 'solution consultant', 'security consultant', 'advisor'],
    }

    for category_id, keywords in category_keywords.items():
        if any(kw in text for kw in keywords):
            return category_id

    return 0

def demo_predict(title, skillset):
    """Demo mode prediction logic"""
    text = f"{title} {skillset}"
    category = classify_job_category(text)

    try:
        if extractor_mode == 'hybrid':
            extraction = skill_extractor.extract_skills(skillset)
            skills = extraction.get("combined_skills", [])
        else:
            skills = skill_extractor.extract_all_skills(skillset)
    except Exception as e:
        logger.warning(f"Skill extraction failed: {e}")
        skills = []

    return {
        'job_category': category,
        'skills': skills
    }

# ==================== 文件处理功能 ====================

def extract_text_from_pdf(file_content):
    """从PDF文件提取文本"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(file_content):
    """从DOCX文件提取文本"""
    try:
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_doc(file_content):
    """从DOC文件提取文本（简单处理）"""
    try:
        # 对于DOC文件，我们进行简单的文本提取
        text = file_content.decode('utf-8', errors='ignore')
        return text.strip()
    except Exception as e:
        logger.error(f"DOC extraction error: {e}")
        return ""

def extract_text_from_txt(file_content):
    """从TXT文件提取文本"""
    try:
        text = file_content.decode('utf-8', errors='ignore')
        return text.strip()
    except Exception as e:
        logger.error(f"TXT extraction error: {e}")
        return ""

def extract_text_from_file(filename, file_content):
    """根据文件类型提取文本"""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(file_content)
    elif filename_lower.endswith('.doc'):
        return extract_text_from_doc(file_content)
    elif filename_lower.endswith('.txt'):
        return extract_text_from_txt(file_content)
    else:
        # 尝试作为文本文件处理
        return extract_text_from_txt(file_content)

def process_single_job(filename, content, job_id):
    """处理单个工作描述"""
    try:
        # 提取文本
        text = extract_text_from_file(filename, content)
        
        if not text:
            return {
                'job_id': job_id,
                'filename': filename,
                'success': False,
                'error': 'Could not extract text from file'
            }
        
        # 尝试从文本中分离标题和描述
        lines = text.split('\n')
        title = lines[0][:100] if lines else "Unknown Position"  # 使用第一行作为标题
        skillset = text
        
        # 使用现有的预测逻辑
        if model_loaded and classifier:
            try:
                result = classifier.predict(title, skillset)
            except Exception as e:
                logger.warning(f"Trained classifier failed for {filename}: {e}")
                result = demo_predict(title, skillset)
        else:
            result = demo_predict(title, skillset)
        
        # 技能提取
        try:
            if extractor_mode == 'hybrid':
                extraction = skill_extractor.extract_skills(skillset)
                skills = extraction.get("combined_skills", [])
            else:
                skills = skill_extractor.extract_all_skills(skillset)
        except Exception as e:
            logger.warning(f"Skill extraction failed for {filename}: {e}")
            skills = []
        
        # 获取类别信息
        category_id = result['job_category']
        category_info = JOB_CATEGORIES.get(category_id, {
            "name": f"Category {category_id}",
            "description": "Unknown category",
            "color": "#757575",
        })
        
        return {
            'job_id': job_id,
            'filename': filename,
            'success': True,
            'title': title,
            'extracted_text_length': len(text),
            'prediction': {
                'category_id': category_id,
                'category_name': category_info['name'],
                'category_description': category_info['description'],
                'category_color': category_info['color']
            },
            'skills': {
                'extracted_skills': skills[:20],
                'skill_count': len(skills)
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return {
            'job_id': job_id,
            'filename': filename,
            'success': False,
            'error': str(e)
        }

# ==================== API路由 ====================

@app.route('/')
def index():
    """Root endpoint - API documentation"""
    return jsonify({
        'success': True,
        'message': 'Job Classification & Skill Extraction API',
        'version': '1.3.0',
        'status': 'running',
        'endpoints': {
            'predict': '/api/predict (POST)',
            'batch_predict': '/api/batch-predict (POST)',
            'explain_skills': '/api/explain-skills (POST)',
            'status': '/api/status (GET)',
            'categories': '/api/categories (GET)',
            'health': '/health (GET)'
        },
        'features': {
            'job_classification': True,
            'skill_extraction': True,
            'batch_processing': True,
            'ai_skill_explanations': skill_explainer.is_available() if skill_explainer else False
        },
        'model_info': {
            'mode': 'trained_model' if model_loaded else 'demo_mode',
            'extractor_mode': extractor_mode,
            'categories_count': len(JOB_CATEGORIES)
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict_job():
    """Single job prediction API"""
    try:
        data = request.get_json()
        
        if not data or 'title' not in data or 'skillset' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: title and skillset'
            }), 400

        title = data['title'].strip()
        skillset = data['skillset'].strip()
        explain_skills = data.get('explain_skills', True)

        if not title or not skillset:
            return jsonify({
                'success': False,
                'error': 'Title and skillset cannot be empty'
            }), 400

        # 1. Job classification
        if model_loaded and classifier:
            try:
                result = classifier.predict(title, skillset)
            except Exception as e:
                logger.warning(f"Trained classifier failed: {e}, falling back to rule-based")
                result = demo_predict(title, skillset)
        else:
            result = demo_predict(title, skillset)

        # 2. Skill extraction
        try:
            if extractor_mode == 'hybrid':
                extraction = skill_extractor.extract_skills(skillset)
                skills = extraction.get("combined_skills", [])
            else:
                skills = skill_extractor.extract_all_skills(skillset)
        except Exception as e:
            logger.warning(f"Skill extraction failed: {e}")
            skills = []

        # 3. Get category information
        category_id = result['job_category']
        category_info = JOB_CATEGORIES.get(category_id, {
            "name": f"Category {category_id}",
            "description": "Unknown category",
            "color": "#757575",
        })

        # 4. Generate AI skill explanations
        skill_explanations = None
        if explain_skills and skill_explainer and skills:
            try:
                explanation_result = skill_explainer.explain_skills(
                    skills=skills[:10],  # Limit to 10 skills
                    job_category=category_info['name']
                )
                skill_explanations = explanation_result
                logger.info(f"Skill explanations generated: {explanation_result.get('success', False)}")
            except Exception as e:
                logger.error(f"Failed to generate skill explanations: {e}")
                skill_explanations = {
                    'success': False,
                    'error': str(e),
                    'skill_explanations': {skill: f"{skill} - Learn more about this technology" for skill in skills[:10]}
                }

        # 5. Combine response
        response_data = {
            'success': True,
            'prediction': {
                'category_id': category_id,
                'category_name': category_info['name'],
                'category_description': category_info['description'],
                'category_color': category_info['color']
            },
            'skills': {
                'extracted_skills': skills[:20],
                'skill_count': len(skills)
            },
            'model_info': {
                'mode': 'trained_model' if model_loaded else 'demo_mode',
                'extractor_mode': extractor_mode,
                'ai_explainer_available': skill_explainer.is_available() if skill_explainer else False,
                'timestamp': datetime.now().isoformat()
            }
        }

        # Add skill explanations (if generated)
        if skill_explanations:
            response_data['skill_explanations'] = skill_explanations

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'Error during prediction: {str(e)}'
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """批量工作分类和技能提取"""
    try:
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files uploaded'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # 检查文件数量限制
        if len(files) > 50:
            return jsonify({
                'success': False,
                'error': 'Maximum 50 files allowed per batch'
            }), 400
        
        # 创建批量处理ID
        batch_id = str(uuid.uuid4())
        
        # 准备处理任务
        processing_tasks = []
        for i, file in enumerate(files):
            if file.filename == '':
                continue
                
            # 检查文件大小
            file.seek(0, 2)  # 移动到文件末尾
            file_size = file.tell()
            file.seek(0)  # 重置到开始
            
            if file_size > 10 * 1024 * 1024:  # 10MB限制
                continue
            
            # 检查文件类型
            allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
            if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                continue
            
            file_content = file.read()
            job_id = f"{batch_id}_{i}"
            
            processing_tasks.append({
                'filename': file.filename,
                'content': file_content,
                'job_id': job_id
            })
        
        if not processing_tasks:
            return jsonify({
                'success': False,
                'error': 'No valid files found to process'
            }), 400
        
        # 并行处理文件
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(process_single_job, task['filename'], task['content'], task['job_id']): task 
                for task in processing_tasks
            }
            
            for future in future_to_task:
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    results.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    results.append({
                        'job_id': task['job_id'],
                        'filename': task['filename'],
                        'success': False,
                        'error': f'Processing timeout or error: {str(e)}'
                    })
        
        # 统计结果
        successful_jobs = [r for r in results if r['success']]
        failed_jobs = [r for r in results if not r['success']]
        
        # 统计技能和类别分布
        all_skills = []
        category_distribution = {}
        
        for job in successful_jobs:
            all_skills.extend(job['skills']['extracted_skills'])
            category_name = job['prediction']['category_name']
            category_distribution[category_name] = category_distribution.get(category_name, 0) + 1
        
        # 计算最常见的技能
        skill_frequency = Counter(all_skills)
        top_skills = skill_frequency.most_common(20)
        
        # 存储批量结果
        with batch_lock:
            batch_results[batch_id] = {
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'total_files': len(files),
                'processed_files': len(processing_tasks),
                'successful_jobs': len(successful_jobs),
                'failed_jobs': len(failed_jobs),
                'results': results,
                'statistics': {
                    'category_distribution': category_distribution,
                    'top_skills': [{'skill': skill, 'count': count} for skill, count in top_skills],
                    'total_unique_skills': len(set(all_skills))
                }
            }
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'summary': {
                'total_files_uploaded': len(files),
                'files_processed': len(processing_tasks),
                'successful_jobs': len(successful_jobs),
                'failed_jobs': len(failed_jobs)
            },
            'results': results,
            'statistics': {
                'category_distribution': category_distribution,
                'top_skills': [{'skill': skill, 'count': count} for skill, count in top_skills[:10]],
                'total_unique_skills': len(set(all_skills))
            },
            'model_info': {
                'mode': 'trained_model' if model_loaded else 'demo_mode',
                'extractor_mode': extractor_mode,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({
            'success': False,
            'error': f'Batch processing failed: {str(e)}'
        }), 500

@app.route('/api/batch-results/<batch_id>', methods=['GET'])
def get_batch_results(batch_id):
    """获取批量处理结果"""
    try:
        with batch_lock:
            if batch_id not in batch_results:
                return jsonify({
                    'success': False,
                    'error': 'Batch ID not found'
                }), 404
            
            return jsonify({
                'success': True,
                'data': batch_results[batch_id]
            })
            
    except Exception as e:
        logger.error(f"Error retrieving batch results: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch-export/<batch_id>', methods=['GET'])
def export_batch_results(batch_id):
    """导出批量处理结果为CSV"""
    try:
        with batch_lock:
            if batch_id not in batch_results:
                return jsonify({
                    'success': False,
                    'error': 'Batch ID not found'
                }), 404
            
            batch_data = batch_results[batch_id]
        
        # 创建CSV数据
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入标题行
        writer.writerow([
            'Filename', 'Job Title', 'Category', 'Category Description', 
            'Skills Count', 'Top Skills', 'Success', 'Error Message'
        ])
        
        # 写入数据行
        for result in batch_data['results']:
            if result['success']:
                skills_list = ', '.join(result['skills']['extracted_skills'][:10])
                writer.writerow([
                    result['filename'],
                    result.get('title', 'N/A'),
                    result['prediction']['category_name'],
                    result['prediction']['category_description'],
                    result['skills']['skill_count'],
                    skills_list,
                    'Yes',
                    ''
                ])
            else:
                writer.writerow([
                    result['filename'],
                    'N/A',
                    'N/A',
                    'N/A',
                    0,
                    '',
                    'No',
                    result.get('error', 'Unknown error')
                ])
        
        # 准备响应
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=batch_results_{batch_id}.csv'
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting batch results: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/explain-skills', methods=['POST'])
def explain_skills_endpoint():
    """Dedicated API endpoint for explaining skills"""
    try:
        data = request.get_json()
        
        if not data or 'skills' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: skills'
            }), 400

        skills = data['skills']
        job_category = data.get('job_category', '')

        if not isinstance(skills, list):
            return jsonify({
                'success': False,
                'error': 'Skills must be provided as a list'
            }), 400

        if not skill_explainer:
            return jsonify({
                'success': False,
                'error': 'AI skill explainer not available'
            }), 503

        # Generate explanations
        explanation_result = skill_explainer.explain_skills(
            skills=skills,
            job_category=job_category
        )

        return jsonify(explanation_result)

    except Exception as e:
        logger.error(f"Skill explanation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all job categories"""
    return jsonify({
        'success': True,
        'categories': JOB_CATEGORIES
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get application status"""
    with batch_lock:
        active_batches = len(batch_results)
    
    return jsonify({
        'success': True,
        'status': {
            'model_loaded': model_loaded,
            'extractor_mode': extractor_mode,
            'ai_explainer_available': skill_explainer.is_available() if skill_explainer else False,
            'categories_count': len(JOB_CATEGORIES),
            'version': '1.3.0',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'mode': 'trained_model' if model_loaded else 'demo_mode',
            'features': {
                'job_classification': True,
                'skill_extraction': True,
                'ai_skill_explanations': skill_explainer.is_available() if skill_explainer else False,
                'batch_processing': True,
                'file_upload': True,
            },
            'batch_processing': {
                'active_batches': active_batches,
                'max_files_per_batch': 50,
                'max_file_size_mb': 10,
                'supported_formats': ['PDF', 'DOCX', 'DOC', 'TXT']
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'extractor_mode': extractor_mode,
        'ai_available': skill_explainer.is_available() if skill_explainer else False
    })

# Error handling
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Requested resource not found',
        'available_endpoints': [
            '/',
            '/api/predict',
            '/api/batch-predict',
            '/api/explain-skills',
            '/api/status',
            '/api/categories',
            '/health'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

# Application startup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render默认端口是10000
    app.run(host="0.0.0.0", port=port)
