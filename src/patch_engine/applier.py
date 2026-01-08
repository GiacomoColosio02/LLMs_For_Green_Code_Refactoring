"""
Patch Engine: Extract and apply patches from LLM responses.
Supports multiple formats: SEARCH/REPLACE blocks, unified diff, git diff.
Includes fuzzy matching for robust application.
"""
import re
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PatchBlock:
    """Represents a single SEARCH/REPLACE block."""
    target_file: Optional[str]
    search_content: str
    replace_content: str
    raw_block: str


@dataclass 
class PatchResult:
    """Result of patch application."""
    success: bool
    changes_applied: int
    total_blocks: int
    method_used: str  # 'search_replace', 'git_apply', 'none'
    error_message: Optional[str] = None
    modified_files: List[str] = None
    
    def __post_init__(self):
        if self.modified_files is None:
            self.modified_files = []


class PatchEngine:
    """
    Extract and apply patches from LLM responses.
    
    Supports:
    - SWE-bench SEARCH/REPLACE format
    - Unified diff format
    - Git diff format
    
    Features:
    - Fuzzy matching (ignores whitespace differences)
    - Automatic file detection
    - Multiple fallback strategies
    """
    
    def __init__(self, repo_path: Path):
        """
        Initialize patch engine.
        
        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)
    
    # =========================================================================
    # MAIN PUBLIC API
    # =========================================================================
    
    def apply_patch(
        self, 
        llm_response: str, 
        candidate_files: Optional[List[str]] = None
    ) -> PatchResult:
        """
        Apply patch from LLM response to repository.
        
        Args:
            llm_response: Raw LLM output containing patch
            candidate_files: Optional list of files that might be modified
                           (helps with file detection)
        
        Returns:
            PatchResult with details about the application
        """
        if candidate_files is None:
            candidate_files = []
        
        # Clean response (remove thinking tags, etc.)
        cleaned = self._clean_response(llm_response)
        
        # Try SEARCH/REPLACE format first (preferred)
        if "<<<<<<< SEARCH" in cleaned:
            result = self._apply_search_replace(cleaned, candidate_files)
            if result.success:
                return result
        
        # Try git apply as fallback
        result = self._apply_git_diff(cleaned)
        if result.success:
            return result
        
        # Nothing worked
        return PatchResult(
            success=False,
            changes_applied=0,
            total_blocks=0,
            method_used='none',
            error_message='Could not apply patch with any method'
        )
    
    def extract_patch_content(self, llm_response: str) -> str:
        """
        Extract just the patch content from LLM response.
        Useful for saving/debugging.
        
        Args:
            llm_response: Raw LLM output
            
        Returns:
            Cleaned patch content
        """
        return self._clean_response(llm_response)
    
    # =========================================================================
    # RESPONSE CLEANING
    # =========================================================================
    
    def _clean_response(self, content: str) -> str:
        """
        Clean LLM response by removing thinking tags and extracting patch.
        
        Args:
            content: Raw LLM response
            
        Returns:
            Cleaned content with just the patch
        """
        # Remove <think>...</think> blocks (DeepSeek R1 style)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # If SEARCH/REPLACE present, extract from there
        if "<<<<<<< SEARCH" in content:
            # Get content starting slightly before first SEARCH block
            idx = content.find("<<<<<<< SEARCH")
            start = max(0, idx - 500)  # Include file path hints
            return content[start:]
        
        # Try to extract from code blocks
        code_blocks = re.findall(r'```(?:diff|python|patch)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks:
            # Return the longest code block (likely the patch)
            return max(code_blocks, key=len).strip()
        
        return content.strip()
    
    # =========================================================================
    # SEARCH/REPLACE APPLICATION
    # =========================================================================
    
    def _apply_search_replace(
        self, 
        content: str, 
        candidate_files: List[str]
    ) -> PatchResult:
        """
        Apply SEARCH/REPLACE blocks to repository.
        
        Args:
            content: Cleaned content with SEARCH/REPLACE blocks
            candidate_files: Files that might be targets
            
        Returns:
            PatchResult
        """
        blocks = self._parse_search_replace_blocks(content)
        
        if not blocks:
            return PatchResult(
                success=False,
                changes_applied=0,
                total_blocks=0,
                method_used='search_replace',
                error_message='No valid SEARCH/REPLACE blocks found'
            )
        
        changes_applied = 0
        modified_files = []
        
        for block in blocks:
            # Try to find target file
            target_file = block.target_file
            
            # If no target file in block, try to detect from candidates
            if not target_file:
                target_file = self._detect_target_file(
                    block.search_content, 
                    candidate_files
                )
            
            if target_file:
                success = self._apply_single_block(
                    target_file,
                    block.search_content,
                    block.replace_content
                )
                if success:
                    changes_applied += 1
                    if target_file not in modified_files:
                        modified_files.append(target_file)
                    continue
            
            # If specific file failed, try all candidates
            for cand in candidate_files:
                if cand == target_file:
                    continue
                success = self._apply_single_block(
                    cand,
                    block.search_content,
                    block.replace_content
                )
                if success:
                    changes_applied += 1
                    if cand not in modified_files:
                        modified_files.append(cand)
                    break
        
        return PatchResult(
            success=changes_applied > 0,
            changes_applied=changes_applied,
            total_blocks=len(blocks),
            method_used='search_replace',
            modified_files=modified_files
        )
    
    def _parse_search_replace_blocks(self, content: str) -> List[PatchBlock]:
        """
        Parse SEARCH/REPLACE blocks from content.
        
        Format:
        ### path/to/file.py  (optional)
        <<<<<<< SEARCH
        ... original code ...
        =======
        ... replacement code ...
        >>>>>>> REPLACE
        
        Args:
            content: Content with SEARCH/REPLACE blocks
            
        Returns:
            List of PatchBlock objects
        """
        blocks = []
        parts = content.split('<<<<<<< SEARCH')
        
        for i, part in enumerate(parts[1:], 1):  # Skip first part (before any block)
            if '=======' not in part or '>>>>>>> REPLACE' not in part:
                continue
            
            try:
                search_part, rest = part.split('=======', 1)
                replace_part, _ = rest.split('>>>>>>> REPLACE', 1)
                
                # Try to find file path from preceding content
                target_file = self._extract_file_path(parts[i-1] if i > 0 else '', content, part)
                
                blocks.append(PatchBlock(
                    target_file=target_file,
                    search_content=search_part.strip('\n'),
                    replace_content=replace_part.strip('\n'),
                    raw_block=f"<<<<<<< SEARCH{part.split('>>>>>>> REPLACE')[0]}>>>>>>> REPLACE"
                ))
            except ValueError:
                continue
        
        return blocks
    
    def _extract_file_path(self, preceding: str, full_content: str, block: str) -> Optional[str]:
        """
        Extract file path from context around a SEARCH/REPLACE block.
        
        Looks for patterns like:
        - ### path/to/file.py
        - File: path/to/file.py
        - `path/to/file.py`
        - path/to/file.py (standalone line ending in .py)
        
        Args:
            preceding: Content before this block
            full_content: Full content (for context)
            block: The current block content
            
        Returns:
            File path or None
        """
        # Check last 30 lines before the block
        lines = preceding.strip().splitlines()[-30:]
        
        for line in reversed(lines):
            # Clean common prefixes
            clean = line.strip()
            clean = re.sub(r'^###\s*', '', clean)
            clean = re.sub(r'^File:\s*', '', clean)
            clean = re.sub(r'^`|`$', '', clean)
            clean = clean.strip()
            
            # Check if it looks like a Python file path
            if clean.endswith('.py') and '/' in clean:
                # Validate it exists in repo
                if (self.repo_path / clean).exists():
                    return clean
                # Try without leading slash
                if clean.startswith('/'):
                    clean = clean[1:]
                    if (self.repo_path / clean).exists():
                        return clean
        
        return None
    
    def _detect_target_file(
        self, 
        search_content: str, 
        candidate_files: List[str]
    ) -> Optional[str]:
        """
        Detect target file by matching search content against candidates.
        
        Args:
            search_content: The SEARCH block content
            candidate_files: List of potential target files
            
        Returns:
            Best matching file path or None
        """
        if not search_content.strip():
            return None
        
        # Get first non-empty line as signature
        lines = [l.strip() for l in search_content.splitlines() if l.strip()]
        if not lines:
            return None
        
        signature = lines[0].replace('"', "'")
        
        # Search in candidates
        for cand in candidate_files:
            path = self.repo_path / cand
            if path.exists():
                try:
                    content = path.read_text(errors='ignore').replace('"', "'")
                    if signature in content:
                        return cand
                except Exception:
                    continue
        
        return None
    
    def _apply_single_block(
        self,
        rel_path: str,
        search_content: str,
        replace_content: str
    ) -> bool:
        """
        Apply a single SEARCH/REPLACE block to a file using fuzzy matching.
        
        Args:
            rel_path: Relative path to file
            search_content: Content to search for
            replace_content: Content to replace with
            
        Returns:
            True if successful
        """
        target_file = self.repo_path / rel_path
        if not target_file.exists():
            return False
        
        try:
            original_lines = target_file.read_text().splitlines(keepends=True)
        except Exception as e:
            logger.warning(f"Could not read {rel_path}: {e}")
            return False
        
        # Normalize search lines (strip whitespace, normalize quotes)
        search_lines = search_content.splitlines()
        norm_search = [l.strip().replace('"', "'") for l in search_lines if l.strip()]
        
        if not norm_search:
            return False
        
        # Build normalized map of file lines
        file_map = []  # List of (normalized_line, original_index)
        for idx, line in enumerate(original_lines):
            stripped = line.strip().replace('"', "'")
            if stripped:
                file_map.append((stripped, idx))
        
        search_len = len(norm_search)
        match_start_idx = -1
        
        # Find matching window
        for i in range(len(file_map) - search_len + 1):
            window = [item[0] for item in file_map[i:i + search_len]]
            if window == norm_search:
                match_start_idx = i
                break
        
        if match_start_idx == -1:
            return False
        
        # Get real line indices
        real_start = file_map[match_start_idx][1]
        real_end = file_map[match_start_idx + search_len - 1][1]
        
        # Prepare replacement lines (preserve newlines)
        replace_lines = replace_content.splitlines()
        final_replace = []
        for line in replace_lines:
            if not line.endswith('\n'):
                line = line + '\n'
            final_replace.append(line)
        
        # Apply replacement
        new_content = original_lines[:real_start] + final_replace + original_lines[real_end + 1:]
        
        try:
            target_file.write_text(''.join(new_content))
            logger.info(f"Applied patch to {rel_path}")
            return True
        except Exception as e:
            logger.error(f"Could not write {rel_path}: {e}")
            return False
    
    # =========================================================================
    # GIT DIFF APPLICATION
    # =========================================================================
    
    def _apply_git_diff(self, content: str) -> PatchResult:
        """
        Apply patch using git apply with various options.
        
        Args:
            content: Patch content (diff format)
            
        Returns:
            PatchResult
        """
        # Write patch to temp file
        patch_file = self.repo_path / "llm_generated.patch"
        
        try:
            patch_file.write_text(content)
            
            # Try different git apply options
            options_to_try = [
                [],  # Default
                ["-p0"],  # No path stripping
                ["--ignore-space-change"],
                ["--ignore-whitespace"],
                ["-p0", "--ignore-space-change"],
                ["-p0", "--ignore-whitespace"],
            ]
            
            for opts in options_to_try:
                cmd = ["git", "apply"] + opts + [str(patch_file)]
                result = subprocess.run(
                    cmd,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"Applied patch with git apply {' '.join(opts)}")
                    return PatchResult(
                        success=True,
                        changes_applied=1,
                        total_blocks=1,
                        method_used='git_apply'
                    )
            
            # All attempts failed
            return PatchResult(
                success=False,
                changes_applied=0,
                total_blocks=1,
                method_used='git_apply',
                error_message='git apply failed with all options'
            )
            
        except Exception as e:
            return PatchResult(
                success=False,
                changes_applied=0,
                total_blocks=0,
                method_used='git_apply',
                error_message=str(e)
            )
        finally:
            # Cleanup
            if patch_file.exists():
                patch_file.unlink()
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_modified_files(self) -> List[str]:
        """
        Get list of modified files in repository (via git status).
        
        Returns:
            List of modified file paths
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            modified = []
            for line in result.stdout.splitlines():
                if line.startswith(' M') or line.startswith('M '):
                    modified.append(line[3:].strip())
            
            return modified
        except Exception:
            return []
    
    def reset_changes(self):
        """Reset all changes in repository (git checkout)."""
        try:
            subprocess.run(
                ["git", "checkout", "."],
                cwd=self.repo_path,
                capture_output=True,
                check=True
            )
            logger.info("Reset repository changes")
        except Exception as e:
            logger.error(f"Could not reset changes: {e}")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def apply_llm_patch(
    repo_path: Path,
    llm_response: str,
    candidate_files: Optional[List[str]] = None
) -> PatchResult:
    """
    Convenience function to apply LLM patch.
    
    Args:
        repo_path: Path to repository
        llm_response: Raw LLM output
        candidate_files: Optional list of candidate files
        
    Returns:
        PatchResult
    """
    engine = PatchEngine(repo_path)
    return engine.apply_patch(llm_response, candidate_files)