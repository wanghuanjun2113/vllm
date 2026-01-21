"""
Chunk Parser for Position-Agnostic Chunk Cache

This module provides functionality to parse prompts containing chunk delimiters
and extract system prompt, chunks, and user question.
"""

import re
from typing import List, Optional, Tuple


# Chunk delimiter pattern: "# #" (with possible surrounding whitespace)
CHUNK_DELIMITER_PATTERN = r'\s*#\s*#\s*'


class ChunkParser:
    """
    Parser for chunk-based prompts using '# #' delimiters.

    The parser extracts:
    - System prompt: Content before the first delimiter
    - Chunks: Content between delimiters
    - User question: Content after the last delimiter

    Example:
        Input: "System prompt # # Chunk1 # # Chunk2 # # Question"
        Output:
            system_prompt: "System prompt "
            chunks: ["Chunk1 ", "Chunk2 "]
            user_question: "Question"
    """

    def __init__(self, delimiter: str = "# #"):
        """
        Initialize the chunk parser.

        Args:
            delimiter: The delimiter string to use for splitting chunks.
                       Default is "# #".
        """
        self.delimiter = delimiter
        # Compile regex pattern for better performance
        self.pattern = re.compile(CHUNK_DELIMITER_PATTERN)

    def has_chunks(self, prompt: str) -> bool:
        """
        Check if the prompt contains chunk delimiters.

        Args:
            prompt: The input prompt string

        Returns:
            True if the prompt contains '# #' delimiters, False otherwise.
        """
        if not prompt:
            return False
        return self.pattern.search(prompt) is not None

    def parse(self, prompt: str) -> "ChunkParseResult":
        """
        Parse a prompt containing chunk delimiters.

        Args:
            prompt: The input prompt string with '# #' delimiters

        Returns:
            ChunkParseResult containing system_prompt, chunks, and user_question

        Raises:
            ValueError: If the prompt format is invalid
        """
        from vllm.v1.engine.chunk_metadata import ChunkParseResult

        if not prompt:
            return ChunkParseResult(
                system_prompt="",
                chunks=[],
                user_question="",
                has_chunks=False
            )

        # Find all delimiter positions
        delimiters = []
        for match in self.pattern.finditer(prompt):
            delimiters.append(match.start())

        # If no delimiters found, return as non-chunked result
        if not delimiters:
            return ChunkParseResult(
                system_prompt=prompt,
                chunks=[],
                user_question="",
                has_chunks=False
            )

        # Extract system prompt (before first delimiter)
        system_prompt = prompt[:delimiters[0]]

        # Extract chunks (between delimiters)
        chunks = []
        for i in range(len(delimiters)):
            start = delimiters[i] + len(match_at_pos(prompt, delimiters[i]))
            if i + 1 < len(delimiters):
                end = delimiters[i + 1]
            else:
                end = len(prompt)
            chunk_content = prompt[start:end].strip()
            if chunk_content:  # Only add non-empty chunks
                chunks.append(chunk_content)

        # Extract user question (after last delimiter)
        last_delimiter_end = delimiters[-1] + len(match_at_pos(prompt, delimiters[-1]))
        user_question = prompt[last_delimiter_end:].strip()

        # Validate that we have at least one chunk
        if not chunks:
            raise ValueError(
                f"Prompt contains delimiters but no valid chunks found. "
                f"Format: 'system_prompt # # chunk1 # # chunk2 # # question'"
            )

        return ChunkParseResult(
            system_prompt=system_prompt.strip(),
            chunks=chunks,
            user_question=user_question,
            delimiters=delimiters,
            has_chunks=True
        )

    def split_into_chunks(
        self,
        prompt: str
    ) -> Tuple[str, List[str], str]:
        """
        Split a prompt into system prompt, chunks, and user question.

        This is a convenience method that returns the three components directly.

        Args:
            prompt: The input prompt string

        Returns:
            Tuple of (system_prompt, chunks_list, user_question)
        """
        result = self.parse(prompt)
        return result.system_prompt, result.chunks, result.user_question

    def validate_format(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the format of a chunked prompt.

        Args:
            prompt: The input prompt string

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if format is valid
            - error_message: Error message if invalid, None otherwise
        """
        try:
            result = self.parse(prompt)

            if not result.has_chunks:
                return True, None

            # Check if we have chunks
            if not result.chunks:
                return False, "No chunks found between delimiters"

            # Check if user question exists (optional, but recommended)
            # We don't enforce this strictly as some use cases may not need it

            return True, None

        except Exception as e:
            return False, str(e)

    def get_chunk_positions(self, prompt: str) -> List[Tuple[int, int]]:
        """
        Get the start and end positions of each chunk in the original prompt.

        Args:
            prompt: The input prompt string

        Returns:
            List of (start_pos, end_pos) tuples for each chunk
        """
        if not self.has_chunks(prompt):
            return []

        delimiters = []
        for match in self.pattern.finditer(prompt):
            delimiters.append((match.start(), match.end()))

        chunk_positions = []
        for i in range(len(delimiters)):
            start = delimiters[i][1]
            if i + 1 < len(delimiters):
                end = delimiters[i + 1][0]
            else:
                end = len(prompt)
            chunk_positions.append((start, end))

        return chunk_positions


def match_at_pos(text: str, pos: int) -> str:
    """
    Helper function to get the matched delimiter at a specific position.

    Args:
        text: The text to search in
        pos: The position to find the match at

    Returns:
        The matched delimiter string
    """
    pattern = re.compile(CHUNK_DELIMITER_PATTERN)
    match = pattern.search(text[pos:])
    if match:
        return match.group()
    return "# #"


def create_chunk_parser(delimiter: str = "# #") -> ChunkParser:
    """
    Factory function to create a ChunkParser instance.

    Args:
        delimiter: The delimiter string to use

    Returns:
        A new ChunkParser instance
    """
    return ChunkParser(delimiter=delimiter)
