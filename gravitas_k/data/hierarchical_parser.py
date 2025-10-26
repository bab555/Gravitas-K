"""
Hierarchical Document Parser for HVAE Training

Parses documents into hierarchical structure:
- Directory level (entire document)
- Page level (chapters/sections)
- Line level (paragraphs)
- Phrase level (sentences)
- Word level (tokens)

This structure aligns with HVAE's latent hierarchy: 768D -> 256D -> 128D -> 64D -> 32D
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass
class HierarchicalDocument:
    """Represents a document with hierarchical structure."""
    
    # Raw text
    raw_text: str
    
    # Hierarchical structure
    directory: str  # Entire document
    pages: List[str]  # Chapters/sections
    lines: List[List[str]]  # Paragraphs within each page
    phrases: List[List[List[str]]]  # Sentences within each line
    words: List[List[List[List[str]]]]  # Words within each phrase
    
    # Offsets for alignment
    page_offsets: List[int]
    line_offsets: List[List[int]]
    phrase_offsets: List[List[List[int]]]
    word_offsets: List[List[List[List[int]]]]


class HierarchicalDocumentParser:
    """
    Parses documents into hierarchical structure for HVAE training.
    
    This parser extracts multi-level structure that maps to HVAE's latent hierarchy:
    - Directory (768D): Global document context
    - Page (256D): Chapter/section-level themes
    - Line (128D): Paragraph-level semantics
    - Phrase (64D): Sentence-level meaning
    - Word (32D): Token-level details
    """
    
    def __init__(
        self,
        min_page_length: int = 500,
        min_line_length: int = 50,
        sentence_split_pattern: str = r'[.!?]+\s+',
        word_tokenizer: Optional[callable] = None,
    ):
        """
        Args:
            min_page_length: Minimum character length to consider as a separate page
            min_line_length: Minimum character length to consider as a separate line
            sentence_split_pattern: Regex pattern for splitting sentences
            word_tokenizer: Optional custom word tokenizer (defaults to whitespace split)
        """
        self.min_page_length = min_page_length
        self.min_line_length = min_line_length
        self.sentence_pattern = re.compile(sentence_split_pattern)
        self.word_tokenizer = word_tokenizer or self._default_word_tokenizer
        
    def _default_word_tokenizer(self, text: str) -> List[str]:
        """Default word tokenizer using whitespace."""
        return text.split()
    
    def parse(self, text: str) -> HierarchicalDocument:
        """
        Parse a document into hierarchical structure.
        
        Args:
            text: Raw document text
            
        Returns:
            HierarchicalDocument with parsed structure and offsets
        """
        # Directory level: entire document
        directory = text.strip()
        
        # Page level: split by double newlines or section markers
        pages, page_offsets = self._split_pages(directory)
        
        # Line level: split each page into paragraphs
        lines = []
        line_offsets = []
        for page in pages:
            page_lines, page_line_offsets = self._split_lines(page)
            lines.append(page_lines)
            line_offsets.append(page_line_offsets)
        
        # Phrase level: split each line into sentences
        phrases = []
        phrase_offsets = []
        for page_lines in lines:
            page_phrases = []
            page_phrase_offsets = []
            for line in page_lines:
                line_phrases, line_phrase_offsets = self._split_sentences(line)
                page_phrases.append(line_phrases)
                page_phrase_offsets.append(line_phrase_offsets)
            phrases.append(page_phrases)
            phrase_offsets.append(page_phrase_offsets)
        
        # Word level: tokenize each phrase
        words = []
        word_offsets = []
        for page_phrases in phrases:
            page_words = []
            page_word_offsets = []
            for line_phrases in page_phrases:
                line_words = []
                line_word_offsets = []
                for phrase in line_phrases:
                    phrase_words = self.word_tokenizer(phrase)
                    phrase_word_offsets = self._compute_word_offsets(phrase, phrase_words)
                    line_words.append(phrase_words)
                    line_word_offsets.append(phrase_word_offsets)
                page_words.append(line_words)
                page_word_offsets.append(line_word_offsets)
            words.append(page_words)
            word_offsets.append(page_word_offsets)
        
        return HierarchicalDocument(
            raw_text=text,
            directory=directory,
            pages=pages,
            lines=lines,
            phrases=phrases,
            words=words,
            page_offsets=page_offsets,
            line_offsets=line_offsets,
            phrase_offsets=phrase_offsets,
            word_offsets=word_offsets,
        )
    
    def _split_pages(self, text: str) -> Tuple[List[str], List[int]]:
        """Split document into pages (chapters/sections)."""
        # Try to detect section headers (e.g., "Chapter 1", "## Section")
        # For now, use double newlines as page boundaries
        pages = []
        offsets = []
        current_offset = 0
        
        # Split by markdown headers or double newlines
        chunks = re.split(r'\n{2,}', text)
        
        current_page = ""
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            if len(current_page) + len(chunk) < self.min_page_length:
                current_page += "\n\n" + chunk if current_page else chunk
            else:
                if current_page:
                    pages.append(current_page)
                    offsets.append(current_offset)
                    current_offset += len(current_page) + 2  # +2 for newlines
                current_page = chunk
        
        # Add last page
        if current_page:
            pages.append(current_page)
            offsets.append(current_offset)
        
        # If no pages were created, treat entire text as one page
        if not pages:
            pages = [text]
            offsets = [0]
        
        return pages, offsets
    
    def _split_lines(self, page: str) -> Tuple[List[str], List[int]]:
        """Split page into lines (paragraphs)."""
        lines = []
        offsets = []
        current_offset = 0
        
        # Split by single newlines
        chunks = page.split('\n')
        
        current_line = ""
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            if len(current_line) + len(chunk) < self.min_line_length:
                current_line += " " + chunk if current_line else chunk
            else:
                if current_line:
                    lines.append(current_line)
                    offsets.append(current_offset)
                    current_offset += len(current_line) + 1
                current_line = chunk
        
        # Add last line
        if current_line:
            lines.append(current_line)
            offsets.append(current_offset)
        
        # If no lines, treat entire page as one line
        if not lines:
            lines = [page]
            offsets = [0]
        
        return lines, offsets
    
    def _split_sentences(self, line: str) -> Tuple[List[str], List[int]]:
        """Split line into sentences (phrases)."""
        # Use regex to split by punctuation
        sentences = self.sentence_pattern.split(line)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            sentences = [line]
        
        # Compute offsets
        offsets = []
        current_offset = 0
        for sent in sentences:
            idx = line.find(sent, current_offset)
            if idx == -1:
                idx = current_offset
            offsets.append(idx)
            current_offset = idx + len(sent)
        
        return sentences, offsets
    
    def _compute_word_offsets(self, phrase: str, words: List[str]) -> List[int]:
        """Compute character offsets for each word in phrase."""
        offsets = []
        current_offset = 0
        
        for word in words:
            idx = phrase.find(word, current_offset)
            if idx == -1:
                idx = current_offset
            offsets.append(idx)
            current_offset = idx + len(word)
        
        return offsets
    
    def create_hvae_training_sample(
        self,
        doc: HierarchicalDocument,
        tokenizer: callable,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Create a training sample for HVAE from hierarchical document.
        
        Args:
            doc: Parsed hierarchical document
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized document
                - attention_mask: Attention mask
                - level_ids: Level identifiers for each token (0=word, 1=phrase, 2=line, 3=page, 4=directory)
                - parent_ids: Parent node IDs for hierarchical VAE
        """
        # Tokenize the entire document
        encoding = tokenizer(
            doc.raw_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
        )
        
        input_ids = torch.tensor(encoding['input_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])
        offset_mapping = encoding['offset_mapping']
        
        # Compute level_ids and parent_ids for each token
        level_ids = torch.zeros(max_length, dtype=torch.long)
        parent_ids = torch.zeros(max_length, dtype=torch.long)
        
        # This is a simplified version; in practice, you'd need more sophisticated alignment
        # For now, we'll use a heuristic based on character offsets
        
        for i, (start, end) in enumerate(offset_mapping):
            if start == end:  # Padding token
                level_ids[i] = -1
                parent_ids[i] = -1
                continue
            
            # Determine which hierarchical level this token belongs to
            # For simplicity, assign based on position in document
            # In a real implementation, you'd align with the parsed structure
            char_offset = start
            
            # Placeholder logic (should be replaced with actual alignment)
            if char_offset < len(doc.raw_text) // 4:
                level_ids[i] = 0  # Word level
            elif char_offset < len(doc.raw_text) // 2:
                level_ids[i] = 1  # Phrase level
            elif char_offset < len(doc.raw_text) * 3 // 4:
                level_ids[i] = 2  # Line level
            else:
                level_ids[i] = 3  # Page level
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'level_ids': level_ids,
            'parent_ids': parent_ids,
            'raw_text': doc.raw_text,
        }

