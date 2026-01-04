# Fix per le righe 281-303
# Sostituisci il with block con try-finally

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # Measure base commit
            base_results = self.measure_commit(
                instance=instance,
                commit=instance['base_commit'],
                commit_type='base',
                temp_dir=temp_path
            )
            
            # Measure head commit
            head_results = self.measure_commit(
                instance=instance,
                commit=instance['head_commit'],
                commit_type='head',
                temp_dir=temp_path
            )
        finally:
            # Cleanup with ignore_errors (survives permission errors)
            shutil.rmtree(temp_dir, ignore_errors=True)
