#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill Explainer using OpenAI API
Explains individual skills to help students understand job requirements
技能解释器 - 帮助学生理解每个技能的含义和要求
"""

import logging
from typing import List, Dict, Optional
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import OpenAI, handle import error gracefully
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI not available - install with: pip install openai")
    OPENAI_AVAILABLE = False

class SkillExplainer:
    """Use OpenAI API to explain individual skills for educational purposes"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key, will use environment variable if not provided
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not installed. Explainer will be disabled.")
            return
            
        if not self.api_key:
            logger.warning("OpenAI API key not found. Explainer will be disabled.")
            return
            
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if explainer is available"""
        return OPENAI_AVAILABLE and self.client is not None
    
    def explain_skills(self, skills: List[str], job_category: str = "", max_retries: int = 2) -> Dict:
        """
        Explain individual skills to help students understand job requirements
        
        Args:
            skills: List of extracted skills
            job_category: Job category name for context
            max_retries: Maximum retry attempts
            
        Returns:
            Dictionary containing skill explanations
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'OpenAI client not available',
                'skill_explanations': self._create_fallback_explanations(skills)
            }
        
        if not skills:
            return {
                'success': True,
                'skill_explanations': {},
                'skills_count': 0,
                'generation_method': 'empty_skills'
            }
        
        try:
            # Build prompt for skill explanations
            prompt = self._build_explanation_prompt(skills, job_category)
            
            # Call OpenAI API
            for attempt in range(max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an educational assistant helping students understand technical skills required for jobs. Provide clear, concise explanations that help students know what each skill means and what they need to learn."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        max_tokens=800,  # Increased for multiple explanations
                        temperature=0.3,  # Lower temperature for more consistent educational content
                        top_p=0.9
                    )
                    
                    explanation_text = response.choices[0].message.content.strip()
                    skill_explanations = self._parse_skill_explanations(explanation_text, skills)
                    
                    return {
                        'success': True,
                        'skill_explanations': skill_explanations,
                        'skills_count': len(skills),
                        'generation_method': 'openai_api',
                        'model_used': 'gpt-4o-mini',
                        'attempt': attempt + 1,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                except Exception as api_error:
                    logger.warning(f"OpenAI API attempt {attempt + 1} failed: {api_error}")
                    if attempt == max_retries:
                        raise api_error
                    continue
        
        except Exception as e:
            logger.error(f"Failed to generate skill explanations: {e}")
            return {
                'success': False,
                'error': str(e),
                'skill_explanations': self._create_fallback_explanations(skills),
                'skills_count': len(skills),
                'generation_method': 'fallback'
            }
    
    def _build_explanation_prompt(self, skills: List[str], job_category: str = "") -> str:
        """Build prompt for skill explanations"""
        skills_text = ", ".join(skills[:15])  # Limit to avoid too long prompt
        
        category_context = f" for a {job_category} position" if job_category else ""
        
        prompt = f"""Please explain each of the following technical skills{category_context} to help students understand what they need to learn. For each skill, provide a brief but informative explanation covering:

1. What it is (definition)
2. What you need to know/learn
3. Why it's important for the job

Skills to explain: {skills_text}

Format your response like this:
**Skill Name**: Brief explanation covering what it is, what students need to learn, and why it's important.

Keep each explanation concise (2-3 sentences) but informative for students who may not be familiar with these technologies.
"""
        return prompt
    
    def _parse_skill_explanations(self, explanation_text: str, original_skills: List[str]) -> Dict[str, str]:
        """Parse the AI response into individual skill explanations"""
        explanations = {}
        
        # Split by lines and look for skill explanations
        lines = explanation_text.split('\n')
        current_skill = None
        current_explanation = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new skill explanation
            if line.startswith('**') and line.endswith('**:') or ':' in line:
                # Save previous skill if exists
                if current_skill and current_explanation:
                    explanations[current_skill] = ' '.join(current_explanation).strip()
                
                # Extract skill name
                if line.startswith('**'):
                    current_skill = line.replace('**', '').replace(':', '').strip().lower()
                else:
                    current_skill = line.split(':')[0].strip().lower()
                
                # Get explanation part
                if ':' in line:
                    explanation_part = line.split(':', 1)[1].strip()
                    current_explanation = [explanation_part] if explanation_part else []
                else:
                    current_explanation = []
            else:
                # Continue current explanation
                if current_skill:
                    current_explanation.append(line)
        
        # Don't forget the last skill
        if current_skill and current_explanation:
            explanations[current_skill] = ' '.join(current_explanation).strip()
        
        # Match explanations to original skills (case-insensitive)
        matched_explanations = {}
        for original_skill in original_skills:
            skill_lower = original_skill.lower()
            
            # Direct match
            if skill_lower in explanations:
                matched_explanations[original_skill] = explanations[skill_lower]
            else:
                # Partial match
                for explained_skill, explanation in explanations.items():
                    if skill_lower in explained_skill or explained_skill in skill_lower:
                        matched_explanations[original_skill] = explanation
                        break
                else:
                    # No match found, create basic explanation
                    matched_explanations[original_skill] = f"{original_skill} is a technology/skill required for this position. Students should research and learn about this technology to meet job requirements."
        
        return matched_explanations
    
    def _create_fallback_explanations(self, skills: List[str]) -> Dict[str, str]:
        """Create fallback explanations when OpenAI is unavailable"""
        explanations = {}
        
        # Basic explanations for common skills
        basic_explanations = {
            'python': 'Python is a popular programming language. Learn Python syntax, data structures, and frameworks like Django or Flask.',
            'java': 'Java is an object-oriented programming language. Learn Java fundamentals, OOP concepts, and frameworks like Spring.',
            'javascript': 'JavaScript is essential for web development. Learn JS fundamentals, DOM manipulation, and frameworks like React or Angular.',
            'aws': 'Amazon Web Services (AWS) is a cloud computing platform. Learn AWS services like EC2, S3, and basic cloud concepts.',
            'docker': 'Docker is a containerization platform. Learn how to create, manage, and deploy containers for applications.',
            'kubernetes': 'Kubernetes orchestrates containerized applications. Learn container orchestration, deployment, and scaling concepts.',
            'penetration testing': 'Penetration testing involves testing system security by simulating attacks. Learn security assessment tools and methodologies.',
            'cybersecurity': 'Cybersecurity protects systems from digital attacks. Learn security principles, threat analysis, and protection strategies.',
            'siem': 'SIEM (Security Information and Event Management) systems monitor security events. Learn log analysis and security monitoring.',
            'sql': 'SQL is used for database management. Learn database queries, data manipulation, and database design principles.',
            'git': 'Git is a version control system. Learn how to track code changes, branching, and collaborative development.',
            'linux': 'Linux is an operating system widely used in servers. Learn command line operations and system administration.',
            'react': 'React is a JavaScript library for building user interfaces. Learn component-based development and state management.',
            'angular': 'Angular is a web application framework. Learn TypeScript, component architecture, and Angular CLI.',
            'django': 'Django is a Python web framework. Learn web development patterns, ORM, and building web applications.',
            'flask': 'Flask is a lightweight Python web framework. Learn web development basics and RESTful API creation.',
            'mongodb': 'MongoDB is a NoSQL database. Learn document-based data storage and database operations.',
            'postgresql': 'PostgreSQL is a relational database system. Learn SQL, database design, and advanced database features.'
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower in basic_explanations:
                explanations[skill] = basic_explanations[skill_lower]
            else:
                # Generic explanation
                explanations[skill] = f"{skill} is a technology/skill required for this position. Research this technology to understand its applications and learn the necessary concepts and tools."
        
        return explanations

# Test function
def test_skill_explainer():
    """Test the skill explainer"""
    print("🧪 Testing SkillExplainer...")
    
    explainer = SkillExplainer()
    
    if not explainer.is_available():
        print("⚠️ OpenAI API not available, testing fallback mode")
    
    # Test data
    test_skills = ["Python", "AWS", "Docker", "penetration testing", "SIEM"]
    
    print(f"Input skills: {test_skills}")
    
    result = explainer.explain_skills(
        skills=test_skills,
        job_category="Cybersecurity Analyst"
    )
    
    print(f"\n📊 Explanation result:")
    print(f"Success: {result['success']}")
    
    if result['success']:
        print(f"Method: {result['generation_method']}")
        print(f"\n📚 Skill Explanations:")
        for skill, explanation in result['skill_explanations'].items():
            print(f"\n🔧 **{skill}**: {explanation}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        if 'skill_explanations' in result:
            print(f"\n📚 Fallback Explanations:")
            for skill, explanation in result['skill_explanations'].items():
                print(f"\n🔧 **{skill}**: {explanation}")

if __name__ == "__main__":
    test_skill_explainer()
