"""
Gradio Interactive Demo for Gravitas-K

Provides an interactive web interface for:
- Testing model inference with custom prompts
- Visualizing internal thinking process (<think> tags)
- Adjusting control parameters (σ, γ, λ, α)
- Exporting high-quality/low-quality samples for DPO training
"""

import torch
import gradio as gr
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


class GravitasKDemo:
    """
    Interactive Gradio demo for Gravitas-K model.
    
    Features:
    - Real-time inference with customizable parameters
    - Internal state visualization
    - Think-tag highlighting
    - Control parameter tuning
    - Sample export for DPO
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        export_dir: str = './exported_samples',
    ):
        """
        Args:
            model: Loaded Gravitas-K model
            tokenizer: Hugging Face tokenizer
            device: Device to run inference on
            export_dir: Directory to save exported samples
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.to(device)
        self.model.eval()
        
        # Storage for generated samples
        self.sample_history = []
    
    def generate_with_think(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        sigma_ctrl: float = 0.5,
        gamma_ctrl: float = 0.5,
        fc_strength: float = 1.0,
        alpha_dual: float = 0.5,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response with internal thinking visualization.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            sigma_ctrl: PRC variance control
            gamma_ctrl: PRC gating control
            fc_strength: FlowingContext strength
            alpha_dual: Dual-stream mixing coefficient
            
        Returns:
            Tuple of (generated_text, internal_states)
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Create meta dict for Arbiter
        meta = {
            'sigma_ctrl': sigma_ctrl,
            'gamma_ctrl': gamma_ctrl,
            'fc_strength': fc_strength,
            'alpha_dual': alpha_dual,
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                meta=meta,  # Pass control parameters
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        
        # Extract internal states (if available)
        internal_states = {
            'generated_text': generated_text,
            'prompt': prompt,
            'params': {
                'temperature': temperature,
                'top_p': top_p,
                'sigma_ctrl': sigma_ctrl,
                'gamma_ctrl': gamma_ctrl,
                'fc_strength': fc_strength,
                'alpha_dual': alpha_dual,
            },
        }
        
        # Store in history
        self.sample_history.append(internal_states)
        
        return generated_text, internal_states
    
    def highlight_think_tags(self, text: str) -> str:
        """
        Highlight <think> tags in HTML for visualization.
        
        Args:
            text: Text with <think> tags
            
        Returns:
            HTML-formatted text with highlighted think sections
        """
        import re
        
        # Replace <think> tags with styled HTML
        highlighted = re.sub(
            r'<think>(.*?)</think>',
            r'<div style="background-color: #ffeb3b; border-left: 4px solid #f57c00; padding: 10px; margin: 10px 0; border-radius: 4px;"><strong>🤔 Internal Thought:</strong><br>\1</div>',
            text,
            flags=re.DOTALL
        )
        
        return highlighted
    
    def export_sample(
        self,
        sample_id: int,
        quality: str,
    ) -> str:
        """
        Export a sample for DPO training.
        
        Args:
            sample_id: Index of sample in history
            quality: 'high' or 'low' quality label
            
        Returns:
            Export status message
        """
        if sample_id >= len(self.sample_history):
            return f"❌ Sample {sample_id} not found!"
        
        sample = self.sample_history[sample_id]
        
        # Create export entry
        export_entry = {
            'prompt': sample['prompt'],
            'response': sample['generated_text'],
            'quality': quality,
            'params': sample['params'],
            'timestamp': sample.get('timestamp', ''),
        }
        
        # Save to file
        export_file = self.export_dir / f"{quality}_quality_samples.jsonl"
        with open(export_file, 'a') as f:
            f.write(json.dumps(export_entry) + '\n')
        
        return f"✓ Sample {sample_id} exported as {quality} quality to {export_file}"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Gravitas-K Interactive Demo", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🧠 Gravitas-K Interactive Demo
            
            Explore the internal reasoning process of Gravitas-K with real-time parameter tuning.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input & Controls")
                    
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=5,
                    )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        max_length = gr.Slider(128, 2048, value=512, step=128, label="Max Length")
                        temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                    
                    with gr.Accordion("Gravitas-K Controls", open=True):
                        sigma_ctrl = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="σ (PRC Variance)")
                        gamma_ctrl = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="γ (PRC Gating)")
                        fc_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="λ (FC Strength)")
                        alpha_dual = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="α (Dual-Stream Mix)")
                    
                    generate_btn = gr.Button("🚀 Generate", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Output & Visualization")
                    
                    output_display = gr.HTML(label="Generated Output")
                    
                    with gr.Accordion("Raw Output", open=False):
                        raw_output = gr.Textbox(label="Raw Text", lines=10)
                    
                    with gr.Accordion("Internal States", open=False):
                        internal_states_json = gr.JSON(label="Internal States")
            
            with gr.Row():
                gr.Markdown("### Sample Export for DPO Training")
                
                with gr.Column():
                    sample_id_input = gr.Number(label="Sample ID", value=0, precision=0)
                    quality_radio = gr.Radio(
                        ["high", "low"],
                        label="Quality Label",
                        value="high",
                    )
                    export_btn = gr.Button("💾 Export Sample")
                    export_status = gr.Textbox(label="Export Status", interactive=False)
            
            # Event handlers
            def generate_handler(*args):
                prompt = args[0]
                params = args[1:]
                
                generated_text, internal_states = self.generate_with_think(prompt, *params)
                
                # Highlight think tags
                highlighted_html = self.highlight_think_tags(generated_text)
                
                return highlighted_html, generated_text, internal_states
            
            generate_btn.click(
                fn=generate_handler,
                inputs=[
                    prompt_input,
                    max_length,
                    temperature,
                    top_p,
                    sigma_ctrl,
                    gamma_ctrl,
                    fc_strength,
                    alpha_dual,
                ],
                outputs=[output_display, raw_output, internal_states_json],
            )
            
            export_btn.click(
                fn=self.export_sample,
                inputs=[sample_id_input, quality_radio],
                outputs=export_status,
            )
            
            gr.Markdown("""
            ---
            ### 📖 How to Use
            
            1. **Enter a prompt** in the text box
            2. **Adjust parameters**:
               - Generation: control randomness and length
               - Gravitas-K: fine-tune internal reasoning mechanisms
            3. **Click Generate** to see the output with highlighted internal thoughts
            4. **Export samples** for DPO training by specifying quality labels
            
            ### 🎨 Think-Tag Legend
            
            Yellow highlighted boxes show the model's internal reasoning process:
            - 🤔 **Arbiter**: Task analysis and planning
            - 💡 **Proposer**: Generating hypotheses
            - 🔍 **Challenger**: Questioning and critique
            - ✓ **Verifier**: Validation and checking
            - 🎯 **Synthesizer**: Combining and concluding
            """)
        
        return demo
    
    def launch(self, share: bool = False, server_port: int = 7860):
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_port: Port to run the server on
        """
        demo = self.create_interface()
        demo.launch(share=share, server_port=server_port)

