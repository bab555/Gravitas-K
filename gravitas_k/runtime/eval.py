"""
Evaluation script for Gravitas-K model.

Implements comprehensive evaluation including:
- Long context QA (Hit@k)
- Noise robustness (ΔP@1)
- Creative generation diversity
- Cognitive efficiency (Score/FLOPs)
- Thinking log coherence
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from gravitas_k.models.gravitas_k_model import GravitasKModel, GravitasKConfig


class GravitasKEvaluator:
    """Comprehensive evaluator for Gravitas-K model."""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = 'cuda',
    ):
        self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        
        # Metrics storage
        self.metrics = defaultdict(list)
        
    def _load_model(self, model_path: str):
        """Load Gravitas-K model from checkpoint."""
        # Load configuration
        config_path = Path(model_path) / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                config = GravitasKConfig(**config_dict)
        else:
            # Use default configuration
            config = GravitasKConfig()
            
        # Load model
        model = GravitasKModel(config)
        
        # Load weights if available
        weights_path = Path(model_path) / 'pytorch_model.bin'
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            
        return model.to(self.device)
        
    def evaluate_long_context_qa(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[str],
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate long context question answering.
        
        Measures Hit@k for retrieving relevant information.
        """
        print("Evaluating long context QA...")
        
        hits = []
        em_scores = []
        f1_scores = []
        
        for question, context, answer in tqdm(zip(questions, contexts, answers)):
            # Prepare input
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                max_length=8192,
                truncation=True,
            ).to(self.device)
            
            # Generate with model
            with torch.no_grad():
                outputs = self.model.base_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=False,
                )
                
            # Decode prediction
            prediction = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )
            
            # Calculate metrics
            hit = self._calculate_hit_at_k(prediction, answer, k)
            em = self._calculate_exact_match(prediction, answer)
            f1 = self._calculate_f1(prediction, answer)
            
            hits.append(hit)
            em_scores.append(em)
            f1_scores.append(f1)
            
        return {
            f'hit@{k}': np.mean(hits),
            'exact_match': np.mean(em_scores),
            'f1_score': np.mean(f1_scores),
        }
        
    def evaluate_noise_robustness(
        self,
        texts: List[str],
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
    ) -> Dict[str, float]:
        """
        Evaluate robustness to noisy inputs.
        
        Measures ΔP@1 (change in top-1 probability) under noise.
        """
        print("Evaluating noise robustness...")
        
        delta_p1_scores = defaultdict(list)
        
        for text in tqdm(texts):
            # Get clean prediction
            clean_logits = self._get_logits(text)
            clean_p1 = F.softmax(clean_logits, dim=-1).max().item()
            
            for noise_level in noise_levels:
                if noise_level == 0:
                    continue
                    
                # Add noise to text
                noisy_text = self._add_noise(text, noise_level)
                
                # Get noisy prediction
                noisy_logits = self._get_logits(noisy_text)
                noisy_p1 = F.softmax(noisy_logits, dim=-1).max().item()
                
                # Calculate delta
                delta = abs(clean_p1 - noisy_p1)
                delta_p1_scores[f'noise_{noise_level}'].append(delta)
                
        return {
            f'delta_p1_{key}': np.mean(values)
            for key, values in delta_p1_scores.items()
        }
        
    def evaluate_creative_generation(
        self,
        prompts: List[str],
        num_samples: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate creative generation diversity.
        
        Measures diversity and quality of generated text.
        """
        print("Evaluating creative generation...")
        
        diversity_scores = []
        
        for prompt in tqdm(prompts):
            generations = []
            
            for _ in range(num_samples):
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.base_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.95,
                    )
                    
                generation = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True,
                )
                generations.append(generation)
                
            # Calculate diversity
            diversity = self._calculate_diversity(generations)
            diversity_scores.append(diversity)
            
        return {
            'generation_diversity': np.mean(diversity_scores),
        }
        
    def evaluate_cognitive_efficiency(
        self,
        tasks: List[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate cognitive efficiency (Score/FLOPs).
        
        Measures performance relative to computational cost.
        """
        print("Evaluating cognitive efficiency...")
        
        scores = []
        flops_list = []
        
        for task in tqdm(tasks):
            # Measure FLOPs
            start_flops = self._estimate_flops()
            
            # Perform task
            score = self._perform_task(task)
            
            # Measure FLOPs
            end_flops = self._estimate_flops()
            flops = end_flops - start_flops
            
            scores.append(score)
            flops_list.append(flops)
            
        # Calculate efficiency
        avg_score = np.mean(scores)
        avg_flops = np.mean(flops_list)
        efficiency = avg_score / (avg_flops / 1e9)  # Score per GFLOPs
        
        return {
            'average_score': avg_score,
            'average_gflops': avg_flops / 1e9,
            'cognitive_efficiency': efficiency,
        }
        
    def evaluate_thinking_coherence(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate coherence of thinking logs.
        
        Measures logical consistency and relevance of <think> content.
        """
        print("Evaluating thinking coherence...")
        
        coherence_scores = []
        relevance_scores = []
        
        for prompt in tqdm(prompts):
            # Generate with thinking logs
            outputs, think_logs = self.model.generate_with_think(
                self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device),
                max_new_tokens=200,
            )
            
            # Analyze thinking logs
            if think_logs:
                coherence = self._analyze_coherence(think_logs)
                relevance = self._analyze_relevance(think_logs, prompt)
                
                coherence_scores.append(coherence)
                relevance_scores.append(relevance)
                
        return {
            'thinking_coherence': np.mean(coherence_scores) if coherence_scores else 0,
            'thinking_relevance': np.mean(relevance_scores) if relevance_scores else 0,
        }
        
    def _get_logits(self, text: str) -> torch.Tensor:
        """Get model logits for text."""
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.base_model(**inputs)
            
        return outputs.logits[0, -1, :]
        
    def _add_noise(self, text: str, noise_level: float) -> str:
        """Add character-level noise to text."""
        import random
        
        chars = list(text)
        num_noise = int(len(chars) * noise_level)
        
        for _ in range(num_noise):
            idx = random.randint(0, len(chars) - 1)
            # Random operation: insert, delete, or substitute
            op = random.choice(['insert', 'delete', 'substitute'])
            
            if op == 'insert':
                chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz '))
            elif op == 'delete' and len(chars) > 1:
                del chars[idx]
            else:  # substitute
                chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz ')
                
        return ''.join(chars)
        
    def _calculate_hit_at_k(self, prediction: str, answer: str, k: int) -> float:
        """Calculate Hit@k metric."""
        # Simplified: check if answer appears in top-k tokens
        pred_tokens = prediction.split()[:k]
        answer_tokens = answer.split()
        
        for token in answer_tokens:
            if token in pred_tokens:
                return 1.0
        return 0.0
        
    def _calculate_exact_match(self, prediction: str, answer: str) -> float:
        """Calculate exact match score."""
        return 1.0 if prediction.strip().lower() == answer.strip().lower() else 0.0
        
    def _calculate_f1(self, prediction: str, answer: str) -> float:
        """Calculate F1 score."""
        pred_tokens = set(prediction.lower().split())
        answer_tokens = set(answer.lower().split())
        
        if not pred_tokens or not answer_tokens:
            return 0.0
            
        intersection = pred_tokens & answer_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(answer_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
        
    def _calculate_diversity(self, generations: List[str]) -> float:
        """Calculate diversity of generated texts."""
        if len(generations) < 2:
            return 0.0
            
        # Use unique n-grams as diversity measure
        all_ngrams = set()
        total_ngrams = 0
        
        for gen in generations:
            tokens = gen.split()
            # Bi-grams
            for i in range(len(tokens) - 1):
                ngram = (tokens[i], tokens[i+1])
                all_ngrams.add(ngram)
                total_ngrams += 1
                
        if total_ngrams == 0:
            return 0.0
            
        return len(all_ngrams) / total_ngrams
        
    def _estimate_flops(self) -> float:
        """Estimate FLOPs (simplified)."""
        # This is a placeholder - actual FLOP counting would be more complex
        return torch.cuda.FloatTensor([0]).element_size() * torch.cuda.get_device_properties(0).total_memory
        
    def _perform_task(self, task: Dict) -> float:
        """Perform a task and return score."""
        # Simplified task performance
        prompt = task.get('prompt', '')
        expected = task.get('expected', '')
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.base_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
            )
            
        prediction = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Simple scoring
        return self._calculate_f1(prediction, expected)
        
    def _analyze_coherence(self, think_logs: str) -> float:
        """Analyze coherence of thinking logs."""
        # Simplified: check for logical flow markers
        coherence_markers = ['therefore', 'because', 'however', 'thus', 'so']
        
        score = 0
        for marker in coherence_markers:
            if marker in think_logs.lower():
                score += 1
                
        return min(score / len(coherence_markers), 1.0)
        
    def _analyze_relevance(self, think_logs: str, prompt: str) -> float:
        """Analyze relevance of thinking logs to prompt."""
        # Simplified: check for keyword overlap
        prompt_keywords = set(prompt.lower().split())
        think_keywords = set(think_logs.lower().split())
        
        if not prompt_keywords:
            return 0.0
            
        overlap = len(prompt_keywords & think_keywords)
        return overlap / len(prompt_keywords)
        
    def run_full_evaluation(self, eval_data_path: str) -> Dict[str, float]:
        """Run complete evaluation suite."""
        print("Running full evaluation...")
        
        # Load evaluation data
        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)
            
        all_metrics = {}
        
        # Long context QA
        if 'long_context_qa' in eval_data:
            qa_data = eval_data['long_context_qa']
            qa_metrics = self.evaluate_long_context_qa(
                qa_data['questions'],
                qa_data['contexts'],
                qa_data['answers'],
            )
            all_metrics.update(qa_metrics)
            
        # Noise robustness
        if 'noise_robustness' in eval_data:
            noise_data = eval_data['noise_robustness']
            noise_metrics = self.evaluate_noise_robustness(noise_data['texts'])
            all_metrics.update(noise_metrics)
            
        # Creative generation
        if 'creative_generation' in eval_data:
            creative_data = eval_data['creative_generation']
            creative_metrics = self.evaluate_creative_generation(creative_data['prompts'])
            all_metrics.update(creative_metrics)
            
        # Cognitive efficiency
        if 'cognitive_tasks' in eval_data:
            efficiency_metrics = self.evaluate_cognitive_efficiency(eval_data['cognitive_tasks'])
            all_metrics.update(efficiency_metrics)
            
        # Thinking coherence
        if 'thinking_prompts' in eval_data:
            thinking_metrics = self.evaluate_thinking_coherence(eval_data['thinking_prompts'])
            all_metrics.update(thinking_metrics)
            
        return all_metrics


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Gravitas-K model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='../Qwen3-8B', help='Path to tokenizer')
    parser.add_argument('--eval_data', type=str, default='./data/eval/eval_data.json', help='Path to evaluation data')
    parser.add_argument('--output', type=str, default='./results/eval_results.json', help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = GravitasKEvaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
    )
    
    # Run evaluation
    metrics = evaluator.run_full_evaluation(args.eval_data)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
        
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
