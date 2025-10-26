"""
Noise Augmentation for Robustness Training

Implements various noise injection strategies to improve model robustness:
- Character-level noise (typos, swaps)
- Word-level noise (deletion, substitution)
- Sentence-level noise (reordering, contradiction)
- Semantic noise (paraphrasing, entity swapping)
"""

import random
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass
class NoisyExample:
    """A training example with noise injected."""
    
    original_text: str
    noisy_text: str
    noise_type: str
    noise_level: float  # 0.0 to 1.0
    noise_positions: List[Tuple[int, int]]  # (start, end) character positions


class NoiseAugmentor:
    """
    Applies various noise augmentation strategies to text data.
    
    This helps train the model to be robust to:
    - Typos and misspellings
    - Word deletions or substitutions
    - Contradictory information
    - Paraphrased content
    """
    
    def __init__(
        self,
        char_noise_prob: float = 0.02,
        word_noise_prob: float = 0.05,
        sentence_noise_prob: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            char_noise_prob: Probability of character-level noise per character
            word_noise_prob: Probability of word-level noise per word
            sentence_noise_prob: Probability of sentence-level noise per sentence
            seed: Random seed for reproducibility
        """
        self.char_noise_prob = char_noise_prob
        self.word_noise_prob = word_noise_prob
        self.sentence_noise_prob = sentence_noise_prob
        
        if seed is not None:
            random.seed(seed)
        
        # Common typo patterns
        self.keyboard_neighbors = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcs', 'e': 'wrfd',
            'f': 'rtgcd', 'g': 'tyhfv', 'h': 'yujgb', 'i': 'uokj', 'j': 'ikmnh',
            'k': 'olmj', 'l': 'pk', 'm': 'njk', 'n': 'bhjm', 'o': 'iplk',
            'p': 'ol', 'q': 'wa', 'r': 'etdf', 's': 'wedxa', 't': 'ryfg',
            'u': 'yihj', 'v': 'cfgb', 'w': 'qesa', 'x': 'zsdc', 'y': 'tugh',
            'z': 'asx',
        }
    
    def apply_char_noise(self, text: str, noise_level: float = None) -> NoisyExample:
        """
        Apply character-level noise (typos, swaps, deletions).
        
        Args:
            text: Original text
            noise_level: Override default noise probability
            
        Returns:
            NoisyExample with character-level noise
        """
        if noise_level is None:
            noise_level = self.char_noise_prob
        
        chars = list(text)
        noise_positions = []
        
        i = 0
        while i < len(chars):
            if chars[i].isalpha() and random.random() < noise_level:
                start = i
                noise_type = random.choice(['swap', 'substitute', 'delete', 'duplicate'])
                
                if noise_type == 'swap' and i < len(chars) - 1:
                    # Swap with next character
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                    noise_positions.append((start, i+2))
                    i += 2
                    
                elif noise_type == 'substitute':
                    # Substitute with keyboard neighbor
                    char_lower = chars[i].lower()
                    if char_lower in self.keyboard_neighbors:
                        replacement = random.choice(self.keyboard_neighbors[char_lower])
                        chars[i] = replacement if chars[i].islower() else replacement.upper()
                    noise_positions.append((start, i+1))
                    i += 1
                    
                elif noise_type == 'delete':
                    # Delete character
                    chars.pop(i)
                    noise_positions.append((start, start))
                    # Don't increment i
                    
                elif noise_type == 'duplicate':
                    # Duplicate character
                    chars.insert(i, chars[i])
                    noise_positions.append((start, i+2))
                    i += 2
            else:
                i += 1
        
        noisy_text = ''.join(chars)
        
        return NoisyExample(
            original_text=text,
            noisy_text=noisy_text,
            noise_type='character',
            noise_level=noise_level,
            noise_positions=noise_positions,
        )
    
    def apply_word_noise(self, text: str, noise_level: float = None) -> NoisyExample:
        """
        Apply word-level noise (deletion, substitution, reordering).
        
        Args:
            text: Original text
            noise_level: Override default noise probability
            
        Returns:
            NoisyExample with word-level noise
        """
        if noise_level is None:
            noise_level = self.word_noise_prob
        
        words = text.split()
        noise_positions = []
        
        i = 0
        while i < len(words):
            if random.random() < noise_level:
                noise_type = random.choice(['delete', 'duplicate', 'swap'])
                
                if noise_type == 'delete':
                    # Delete word
                    words.pop(i)
                    noise_positions.append((i, i))
                    # Don't increment i
                    
                elif noise_type == 'duplicate':
                    # Duplicate word
                    words.insert(i, words[i])
                    noise_positions.append((i, i+2))
                    i += 2
                    
                elif noise_type == 'swap' and i < len(words) - 1:
                    # Swap with next word
                    words[i], words[i+1] = words[i+1], words[i]
                    noise_positions.append((i, i+2))
                    i += 2
            else:
                i += 1
        
        noisy_text = ' '.join(words)
        
        return NoisyExample(
            original_text=text,
            noisy_text=noisy_text,
            noise_type='word',
            noise_level=noise_level,
            noise_positions=noise_positions,
        )
    
    def apply_sentence_noise(self, text: str, noise_level: float = None) -> NoisyExample:
        """
        Apply sentence-level noise (reordering, contradiction injection).
        
        Args:
            text: Original text
            noise_level: Override default noise probability
            
        Returns:
            NoisyExample with sentence-level noise
        """
        if noise_level is None:
            noise_level = self.sentence_noise_prob
        
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return NoisyExample(
                original_text=text,
                noisy_text=text,
                noise_type='sentence',
                noise_level=0.0,
                noise_positions=[],
            )
        
        noise_positions = []
        
        if random.random() < noise_level:
            noise_type = random.choice(['reorder', 'contradict'])
            
            if noise_type == 'reorder':
                # Randomly shuffle 2-3 consecutive sentences
                if len(sentences) >= 3:
                    start_idx = random.randint(0, len(sentences) - 3)
                    chunk = sentences[start_idx:start_idx+3]
                    random.shuffle(chunk)
                    sentences[start_idx:start_idx+3] = chunk
                    noise_positions.append((start_idx, start_idx+3))
                    
            elif noise_type == 'contradict':
                # Inject a contradictory statement
                # This is a simplified version; in practice, you'd use an LLM to generate contradictions
                target_idx = random.randint(0, len(sentences) - 1)
                original_sentence = sentences[target_idx]
                
                # Simple contradiction: add "not" or remove "not"
                if ' not ' in original_sentence.lower():
                    contradicted = original_sentence.replace(' not ', ' ', 1).replace(' Not ', ' ', 1)
                else:
                    # Insert "not" before the main verb (simplified heuristic)
                    words = original_sentence.split()
                    if len(words) > 2:
                        # Try to find a verb position (simplified)
                        for i, word in enumerate(words):
                            if word.lower() in ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will', 'would']:
                                words.insert(i+1, 'not')
                                break
                        else:
                            # Fallback: insert after second word
                            words.insert(2, 'not')
                        contradicted = ' '.join(words)
                    else:
                        contradicted = original_sentence
                
                sentences.insert(target_idx + 1, contradicted)
                noise_positions.append((target_idx, target_idx+2))
        
        noisy_text = '. '.join(sentences) + '.'
        
        return NoisyExample(
            original_text=text,
            noisy_text=noisy_text,
            noise_type='sentence',
            noise_level=noise_level,
            noise_positions=noise_positions,
        )
    
    def apply_mixed_noise(
        self,
        text: str,
        char_level: float = None,
        word_level: float = None,
        sentence_level: float = None,
    ) -> NoisyExample:
        """
        Apply a mix of all noise types.
        
        Args:
            text: Original text
            char_level: Character noise level
            word_level: Word noise level
            sentence_level: Sentence noise level
            
        Returns:
            NoisyExample with mixed noise
        """
        # Apply in order: sentence -> word -> character
        result = self.apply_sentence_noise(text, sentence_level)
        result = self.apply_word_noise(result.noisy_text, word_level)
        result = self.apply_char_noise(result.noisy_text, char_level)
        
        result.original_text = text  # Keep original text reference
        result.noise_type = 'mixed'
        
        return result
    
    def create_robustness_training_batch(
        self,
        clean_texts: List[str],
        tokenizer: callable,
        max_length: int = 512,
        noise_ratio: float = 0.5,  # Fraction of examples to apply noise to
    ) -> Dict[str, torch.Tensor]:
        """
        Create a training batch with noise augmentation.
        
        Args:
            clean_texts: List of clean text examples
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            noise_ratio: Fraction of examples to apply noise to
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized texts (mix of clean and noisy)
                - attention_mask: Attention mask
                - labels: Clean labels for all examples
                - noise_mask: Binary mask indicating which examples are noisy
        """
        batch_size = len(clean_texts)
        num_noisy = int(batch_size * noise_ratio)
        
        # Randomly select which examples to make noisy
        noisy_indices = set(random.sample(range(batch_size), num_noisy))
        
        # Create noisy versions
        input_texts = []
        noise_mask = torch.zeros(batch_size, dtype=torch.bool)
        
        for i, text in enumerate(clean_texts):
            if i in noisy_indices:
                noisy_example = self.apply_mixed_noise(text)
                input_texts.append(noisy_example.noisy_text)
                noise_mask[i] = True
            else:
                input_texts.append(text)
        
        # Tokenize inputs (noisy or clean)
        input_encodings = tokenizer(
            input_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize clean targets
        label_encodings = tokenizer(
            clean_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']
        labels = label_encodings['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'noise_mask': noise_mask,
        }

