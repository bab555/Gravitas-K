"""
Think-Tag Alignment Data Generator

Generates training data for aligning CCB internal states with <think> tokens.
This enables the model to learn the A-P-C-V-S workflow through explicit reasoning traces.
"""

import re
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass
class ThinkAlignmentSample:
    """A training sample with think-tag alignment."""
    
    prompt: str
    response_with_think: str  # Response with <think>...</think> tags
    response_without_think: str  # Clean response
    think_segments: List[Dict[str, str]]  # Extracted think segments with labels
    apcvs_labels: Optional[List[str]] = None  # Soft labels for A-P-C-V-S routing


class ThinkAlignmentDataGenerator:
    """
    Generates training data for think-tag alignment.
    
    This generator creates or processes data that contains <think>...</think> tags,
    which represent the model's internal reasoning process. This aligns with
    Qwen3's native thinking mode and Gravitas-K's CCB workflow.
    """
    
    def __init__(
        self,
        think_template: str = "<think>{content}</think>",
        workflow_labels: List[str] = ['arbiter', 'proposer', 'challenger', 'verifier', 'synthesizer'],
    ):
        """
        Args:
            think_template: Template for think tags
            workflow_labels: Labels for A-P-C-V-S workflow stages
        """
        self.think_template = think_template
        self.workflow_labels = workflow_labels
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    
    def parse_think_tags(self, text: str) -> List[Dict[str, str]]:
        """
        Extract think segments from text.
        
        Args:
            text: Text containing <think> tags
            
        Returns:
            List of dictionaries containing think segments and their positions
        """
        segments = []
        for match in self.think_pattern.finditer(text):
            content = match.group(1).strip()
            start = match.start()
            end = match.end()
            
            # Heuristically assign workflow labels based on content
            label = self._infer_workflow_label(content)
            
            segments.append({
                'content': content,
                'start': start,
                'end': end,
                'label': label,
            })
        
        return segments
    
    def _infer_workflow_label(self, content: str) -> str:
        """
        Infer which A-P-C-V-S stage this think segment belongs to.
        
        This is a heuristic approach based on keywords and patterns.
        In practice, you might want to use a classifier or LLM to label these.
        """
        content_lower = content.lower()
        
        # Arbiter: task analysis, planning
        if any(kw in content_lower for kw in ['task', 'plan', 'approach', 'strategy', 'analyze']):
            return 'arbiter'
        
        # Proposer: generating hypotheses, proposals
        elif any(kw in content_lower for kw in ['propose', 'suggest', 'idea', 'hypothesis', 'could be']):
            return 'proposer'
        
        # Challenger: questioning, critiquing
        elif any(kw in content_lower for kw in ['however', 'but', 'challenge', 'question', 'doubt', 'alternative']):
            return 'challenger'
        
        # Verifier: checking, validating
        elif any(kw in content_lower for kw in ['verify', 'check', 'confirm', 'validate', 'correct', 'incorrect']):
            return 'verifier'
        
        # Synthesizer: combining, concluding
        elif any(kw in content_lower for kw in ['therefore', 'thus', 'conclude', 'overall', 'combine', 'final']):
            return 'synthesizer'
        
        # Default to proposer if no clear match
        return 'proposer'
    
    def remove_think_tags(self, text: str) -> str:
        """Remove all <think> tags from text."""
        return self.think_pattern.sub('', text).strip()
    
    def create_alignment_sample(
        self,
        prompt: str,
        response_with_think: str,
    ) -> ThinkAlignmentSample:
        """
        Create an alignment sample from a prompt and response with think tags.
        
        Args:
            prompt: Input prompt
            response_with_think: Response containing <think> tags
            
        Returns:
            ThinkAlignmentSample with parsed segments
        """
        think_segments = self.parse_think_tags(response_with_think)
        response_without_think = self.remove_think_tags(response_with_think)
        
        # Extract A-P-C-V-S labels
        apcvs_labels = [seg['label'] for seg in think_segments]
        
        return ThinkAlignmentSample(
            prompt=prompt,
            response_with_think=response_with_think,
            response_without_think=response_without_think,
            think_segments=think_segments,
            apcvs_labels=apcvs_labels,
        )
    
    def generate_synthetic_think_trace(
        self,
        prompt: str,
        response: str,
        num_think_steps: int = 3,
    ) -> str:
        """
        Generate synthetic <think> tags for a response that doesn't have them.
        
        This is useful for augmenting existing datasets with think traces.
        
        Args:
            prompt: Input prompt
            response: Response without think tags
            num_think_steps: Number of think steps to inject
            
        Returns:
            Response with synthetic <think> tags
        """
        # Split response into sentences
        sentences = re.split(r'[.!?]+\s+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < num_think_steps:
            num_think_steps = len(sentences)
        
        # Inject think tags at strategic points
        think_positions = random.sample(range(len(sentences)), num_think_steps)
        think_positions.sort()
        
        synthetic_response = ""
        last_pos = 0
        
        for i, pos in enumerate(think_positions):
            # Add text before this think step
            synthetic_response += ' '.join(sentences[last_pos:pos]) + '. '
            
            # Generate synthetic think content
            think_content = self._generate_synthetic_think_content(
                prompt, 
                sentences[pos],
                stage=i % len(self.workflow_labels)
            )
            
            synthetic_response += self.think_template.format(content=think_content) + ' '
            last_pos = pos
        
        # Add remaining text
        synthetic_response += ' '.join(sentences[last_pos:])
        
        return synthetic_response.strip()
    
    def _generate_synthetic_think_content(
        self,
        prompt: str,
        current_sentence: str,
        stage: int,
    ) -> str:
        """Generate synthetic think content based on workflow stage."""
        
        workflow_stage = self.workflow_labels[stage]
        
        templates = {
            'arbiter': f"Let me analyze this task. The question asks about: {current_sentence[:50]}...",
            'proposer': f"I propose that {current_sentence[:50]}...",
            'challenger': f"However, I should consider alternative perspectives on {current_sentence[:30]}...",
            'verifier': f"Let me verify this claim: {current_sentence[:50]}...",
            'synthesizer': f"Combining the above reasoning, {current_sentence[:50]}...",
        }
        
        return templates.get(workflow_stage, f"Thinking: {current_sentence[:50]}...")
    
    def create_training_batch(
        self,
        samples: List[ThinkAlignmentSample],
        tokenizer: callable,
        max_length: int = 2048,
    ) -> Dict[str, torch.Tensor]:
        """
        Create a training batch from alignment samples.
        
        Args:
            samples: List of ThinkAlignmentSample
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized prompts + responses
                - attention_mask: Attention mask
                - labels: Language modeling labels
                - think_mask: Binary mask indicating think tokens
                - workflow_labels: Soft labels for A-P-C-V-S routing
        """
        batch_size = len(samples)
        
        # Tokenize prompts and responses
        full_texts = [
            f"{sample.prompt}\n\n{sample.response_with_think}"
            for sample in samples
        ]
        
        encodings = tokenizer(
            full_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # Create labels (same as input_ids for language modeling)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        # Create think_mask (1 for tokens inside <think> tags, 0 otherwise)
        think_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Create workflow_labels (one-hot encoding for A-P-C-V-S)
        num_workflow_stages = len(self.workflow_labels)
        workflow_labels = torch.zeros(batch_size, max_length, num_workflow_stages)
        
        # For each sample, mark think tokens
        for i, sample in enumerate(samples):
            full_text = full_texts[i]
            
            # Find think segments in the tokenized sequence
            for segment in sample.think_segments:
                # Find token positions corresponding to this think segment
                # This is a simplified approach; in practice, you'd need more precise alignment
                segment_start = full_text.find('<think>')
                segment_end = full_text.find('</think>') + len('</think>')
                
                if segment_start == -1 or segment_end == -1:
                    continue
                
                # Tokenize the portion up to segment_start to find token position
                prefix_encoding = tokenizer(
                    full_text[:segment_start],
                    add_special_tokens=False,
                )
                start_token_pos = len(prefix_encoding['input_ids'])
                
                # Tokenize the segment to find its length in tokens
                segment_encoding = tokenizer(
                    full_text[segment_start:segment_end],
                    add_special_tokens=False,
                )
                segment_token_len = len(segment_encoding['input_ids'])
                
                # Mark these tokens as think tokens
                end_token_pos = min(start_token_pos + segment_token_len, max_length)
                think_mask[i, start_token_pos:end_token_pos] = True
                
                # Assign workflow label
                label_idx = self.workflow_labels.index(segment['label'])
                workflow_labels[i, start_token_pos:end_token_pos, label_idx] = 1.0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'think_mask': think_mask,
            'workflow_labels': workflow_labels,
        }

