#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Skill Extractor using spaCy TextRank + Rule-based extraction
Combines graph-based ranking with comprehensive rule-based matching
"""

import logging
logger = logging.getLogger(__name__)

try:
    import spacy
    import pytextrank
    from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"spaCy or pytextrank not available: {e}")
    SPACY_AVAILABLE = False

from demo import BasicSkillExtractor

class HybridSkillExtractor:
    """
    Hybrid skill extractor combining rule-based and graph-based approaches
    """
    
    def __init__(self):
        # Initialize the rule-based keyword extractor
        self.rule_based = BasicSkillExtractor()
        
        # Initialize spaCy components if available
        self.spacy_available = SPACY_AVAILABLE
        self.nlp = None
        
        if self.spacy_available:
            try:
                # Load spaCy's English language model
                self.nlp = spacy.load("en_core_web_sm")
                
                # Create custom stopwords set (from spaCy + domain-specific additions)
                self.custom_stopwords = set(SPACY_STOP_WORDS)
                additional_stopwords = {
                    "we", "are", "have", "experience", "looking", "someone", "requirement",
                    "preferred", "candidate", "you", "skills", "background", "ability", 
                    "knowledge", "years", "strong", "good", "excellent", "proven"
                }
                self.custom_stopwords.update(additional_stopwords)
                
                # Mark additional words as stopwords in spaCy's vocabulary
                for word in additional_stopwords:
                    lex = self.nlp.vocab[word]
                    lex.is_stop = True
                
                # Add TextRank component to the pipeline
                self.nlp.add_pipe("textrank")
                
                logger.info("HybridSkillExtractor initialized with spaCy + TextRank")
                
            except Exception as e:
                logger.warning(f"Failed to initialize spaCy components: {e}")
                self.spacy_available = False
                self.nlp = None
        
        if not self.spacy_available:
            logger.info("HybridSkillExtractor falling back to rule-based extraction only")

    def extract_all_skills(self, text):
        """
        Extract all skills using the hybrid approach - compatibility method
        This method provides the same interface as BasicSkillExtractor.extract_all_skills()
        
        Args:
            text: Job description text
            
        Returns:
            List of extracted skills (combined from rule-based and TextRank)
        """
        extraction_result = self.extract_skills(text)
        return extraction_result.get("combined_skills", [])

    def extract_skills(self, text, topn_textrank=15):
        """
        Extract skill-related phrases from job text using both rule-based
        matching and graph-based ranking (TextRank).
        
        Args:
            text: Job description text
            topn_textrank: Number of top TextRank phrases to consider
            
        Returns:
            Dictionary with rule-based skills, TextRank phrases, and combined results
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "rule_based_skills": [],
                "textrank_phrases": [],
                "combined_skills": []
            }
        
        # 1. Use rule-based extractor (always available)
        rule_skills = set(self.rule_based.extract_all_skills(text))
        
        # 2. Use TextRank extractor with stopword filtering (if available)
        textrank_skills = set()
        
        if self.spacy_available and self.nlp:
            try:
                doc = self.nlp(text)
                
                for phrase in doc._.phrases[:topn_textrank]:
                    cleaned_phrase = phrase.text.strip().lower()
                    
                    # Filter out very short or very long phrases, or phrases made only of stopwords
                    if self._is_valid_textrank_phrase(cleaned_phrase):
                        textrank_skills.add(cleaned_phrase)
                        
            except Exception as e:
                logger.warning(f"TextRank extraction failed: {e}")
                textrank_skills = set()
        
        # 3. Merge rule-based and graph-based results
        combined = sorted(rule_skills.union(textrank_skills))
        
        return {
            "rule_based_skills": sorted(rule_skills),
            "textrank_phrases": sorted(textrank_skills),
            "combined_skills": combined
        }
    
    def _is_valid_textrank_phrase(self, phrase):
        """
        Check if a TextRank phrase is valid for skill extraction
        
        Args:
            phrase: Cleaned phrase text
            
        Returns:
            bool: True if phrase is valid
        """
        if not phrase or len(phrase) < 2:
            return False
        
        # Length constraints
        if len(phrase) > 50 or len(phrase.split()) > 5:
            return False
        
        # Check if phrase is made only of stopwords
        if self.nlp:
            try:
                phrase_doc = self.nlp(phrase)
                if all(token.text.lower() in self.custom_stopwords for token in phrase_doc):
                    return False
            except:
                pass
        
        # Filter out common non-skill phrases
        invalid_phrases = {
            'looking for', 'we are', 'you will', 'able to', 'work with',
            'experience in', 'knowledge of', 'skills in', 'years of',
            'strong understanding', 'good knowledge', 'excellent skills'
        }
        
        if phrase in invalid_phrases:
            return False
        
        return True
    
    def is_spacy_available(self):
        """Check if spaCy components are available"""
        return self.spacy_available
    
    def get_extraction_info(self):
        """Get information about extraction capabilities"""
        return {
            'rule_based_available': True,
            'textrank_available': self.spacy_available,
            'extraction_mode': 'hybrid' if self.spacy_available else 'rule_based_only'
        }

# Test function
def test_hybrid_extractor():
    """Test the HybridSkillExtractor"""
    extractor = HybridSkillExtractor()
    
    test_text = """
    We are looking for a Senior Security Engineer with 5+ years of experience.
    The ideal candidate will have strong expertise in penetration testing,
    vulnerability assessment, and incident response. Technical requirements include:
    Python programming, AWS security, Docker containers, Kubernetes orchestration,
    SIEM tools like Splunk, network security tools including Burp Suite and Nmap.
    Strong communication skills and ability to work in a team environment required.
    """
    
    print("🧪 Testing HybridSkillExtractor")
    print("=" * 50)
    print(f"spaCy Available: {extractor.is_spacy_available()}")
    print(f"Extraction Info: {extractor.get_extraction_info()}")
    
    # Test both methods
    result = extractor.extract_skills(test_text)
    all_skills = extractor.extract_all_skills(test_text)
    
    print(f"\n📊 extract_skills() Results:")
    print(f"Rule-based skills ({len(result['rule_based_skills'])}): {result['rule_based_skills']}")
    print(f"TextRank phrases ({len(result['textrank_phrases'])}): {result['textrank_phrases']}")
    print(f"Combined skills ({len(result['combined_skills'])}): {result['combined_skills']}")
    
    print(f"\n📊 extract_all_skills() Results:")
    print(f"All skills ({len(all_skills)}): {all_skills}")

if __name__ == "__main__":
    test_hybrid_extractor()
