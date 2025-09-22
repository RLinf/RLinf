import re
from typing import List


def qa_accuracy_reward(completions, answers) -> List[float]:
    """
    Reward function that evaluates question-answering accuracy for VQA tasks.
    
    Based on TRL's accuracy_reward pattern but adapted for multiple choice VQA.
    
    Args:
        completions: List of model completions (text strings)
        answers: List of correct answers (text strings)
        
    Returns:
        List of reward scores (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    
    for completion, answer in zip(completions, answers):
        completion_text = str(completion).strip()
        
        # Extract answer from completion - look for <answer>X. content</answer>
        patterns = [
            r'<answer>\s*[A-E]\.\s*(.*?)\s*</answer>',
            r'<answer>\s*[A-E]\s*(.*?)\s*</answer>',
            r'<answer>\s*(.*?)\s*</answer>',
        ]
        
        answer_match = None
        for pattern in patterns:
            answer_match = re.search(pattern, completion_text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                break
        
        if not answer_match:
            rewards.append(0.0)
            continue
            
        predicted_content = answer_match.group(1).strip()
        
        content_match = _compare_choice_content(predicted_content, answer)
        
        rewards.append(1.0 if content_match else 0.0)
    
    return rewards


def _compare_choice_content(predicted: str, correct: str) -> bool:
    """Compare predicted choice content with correct content."""
    # Simple normalized comparison
    pred_normalized = predicted.lower().strip()
    correct_normalized = correct.lower().strip()
    
    # Direct match
    if pred_normalized == correct_normalized:
        return True
    
    # Partial match for more flexibility
    if pred_normalized in correct_normalized or correct_normalized in pred_normalized:
        return True
    
    return False