#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?µÌï© ?êÎèô???åÏù¥?ÑÎùº???§Ï??§Ìä∏?àÏù¥??

?∞Ïù¥??Í∞êÏ?Î∂Ä??Î≤°ÌÑ∞ ?ÑÎ≤†?©ÍπåÏßÄ ?ÑÏ≤¥ ?åÏù¥?ÑÎùº?∏ÏùÑ ?êÎèô?îÌïò???úÏä§?úÏûÖ?àÎã§.
Í∞??®Í≥ÑÎ≥ÑÎ°ú ÏßÑÌñâ ?ÅÌô©??Ï∂îÏ†Å?òÍ≥† Ï≤¥ÌÅ¨?¨Ïù∏?∏Î? Í¥ÄÎ¶¨Ìï©?àÎã§.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse
from dataclasses import dataclass
from tqdm import tqdm

# ?ÑÎ°ú?ùÌä∏ Î£®Ìä∏Î•?Python Í≤ΩÎ°ú??Ï∂îÍ?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.data_processing.auto_data_detector import AutoDataDetector
from scripts.data_processing.incremental_preprocessor import IncrementalPreprocessor
from scripts.data_processing.incremental_precedent_preprocessor import IncrementalPrecedentPreprocessor
from scripts.ml_training.vector_embedding.incremental_vector_builder import IncrementalVectorBuilder
from scripts.ml_training.vector_embedding.incremental_precedent_vector_builder import IncrementalPrecedentVectorBuilder
from scripts.data_processing.utilities.import_laws_to_db import AssemblyLawImporter
from scripts.data_processing.utilities.import_precedents_to_db import PrecedentDataImporter
from scripts.data_collection.common.checkpoint_manager import CheckpointManager
from source.data.database import DatabaseManager

# Import quality modules
try:
    from scripts.data_processing.quality.data_quality_validator import DataQualityValidator
    from scripts.data_processing.quality.automated_data_cleaner import AutomatedDataCleaner
    from scripts.data_processing.quality.real_time_quality_monitor import RealTimeQualityMonitor
    QUALITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Quality modules not available: {e}")
    QUALITY_MODULES_AVAILABLE = False


@dataclass
class PipelineResult:
    """?åÏù¥?ÑÎùº???§Ìñâ Í≤∞Í≥º ?∞Ïù¥???¥Îûò??""
    success: bool
    total_files_detected: int
    files_processed: int
    vectors_added: int
    laws_imported: int
    processing_time: float
    stage_results: Dict[str, Any]
    error_messages: List[str]
    quality_metrics: Dict[str, Any] = None
    quality_improvements: int = 0
    duplicates_resolved: int = 0


class AutoPipelineOrchestrator:
    """?êÎèô???åÏù¥?ÑÎùº???§Ï??§Ìä∏?àÏù¥??""
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 checkpoint_dir: str = "data/checkpoints",
                 db_path: str = "data/lawfirm.db"):
        """
        ?åÏù¥?ÑÎùº???§Ï??§Ìä∏?àÏù¥??Ï¥àÍ∏∞??
        
        Args:
            config: ?åÏù¥?ÑÎùº???§Ï†ï
            checkpoint_dir: Ï≤¥ÌÅ¨?¨Ïù∏???îÎ†â?†Î¶¨
            db_path: ?∞Ïù¥?∞Î≤†?¥Ïä§ Í≤ΩÎ°ú
        """
        self.config = config or self._get_default_config()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Ïª¥Ìè¨?åÌä∏ Ï¥àÍ∏∞??
        self.db_manager = DatabaseManager(db_path)
        self.checkpoint_manager = CheckpointManager(str(self.checkpoint_dir))
        
        # Í∞??®Í≥ÑÎ≥?Ïª¥Ìè¨?åÌä∏
        self.data_detector = AutoDataDetector("data/raw/assembly", self.db_manager)
        self.preprocessor = IncrementalPreprocessor(
            checkpoint_manager=self.checkpoint_manager,
            db_manager=self.db_manager,
            batch_size=self.config['preprocessing']['batch_size']
        )
        self.precedent_preprocessor = IncrementalPrecedentPreprocessor(
            checkpoint_manager=self.checkpoint_manager,
            db_manager=self.db_manager,
            batch_size=self.config['preprocessing']['batch_size']
        )
        self.vector_builder = IncrementalVectorBuilder(
            model_name=self.config['vectorization']['model_name'],
            dimension=self.config['vectorization']['dimension'],
            index_type=self.config['vectorization']['index_type']
        )
        self.precedent_vector_builder = IncrementalPrecedentVectorBuilder(
            model_name=self.config['vectorization']['model_name'],
            dimension=self.config['vectorization']['dimension'],
            index_type=self.config['vectorization']['index_type']
        )
        self.db_importer = AssemblyLawImporter(db_path)
        self.precedent_db_importer = PrecedentDataImporter(db_path)
        
        # ?àÏßà Í¥ÄÎ¶?Ïª¥Ìè¨?åÌä∏ Ï¥àÍ∏∞??
        if QUALITY_MODULES_AVAILABLE:
            self.quality_validator = DataQualityValidator()
            self.data_cleaner = AutomatedDataCleaner(db_path, self.config.get('quality', {}))
            self.quality_monitor = RealTimeQualityMonitor(db_path, self.config.get('quality_monitor', {}))
        else:
            self.quality_validator = None
            self.data_cleaner = None
            self.quality_monitor = None
        
        # Î°úÍπÖ ?§Ï†ï
        self.logger = logging.getLogger(__name__)
        
        # ?åÏù¥?ÑÎùº???ÅÌÉú
        self.pipeline_state = {
            'current_stage': None,
            'start_time': None,
            'end_time': None,
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'stage_results': {}
        }
        
        self.logger.info("AutoPipelineOrchestrator initialized")
    
    def run_auto_pipeline(self, 
                         data_source: str = "law_only",
                         auto_detect: bool = True,
                         specific_path: str = None) -> PipelineResult:
        """
        ?ÑÏ≤¥ ?êÎèô???åÏù¥?ÑÎùº???§Ìñâ
        
        Args:
            data_source: ?∞Ïù¥???åÏä§ ?†Ìòï
            auto_detect: ?êÎèô Í∞êÏ? ?¨Î?
            specific_path: ?πÏ†ï Í≤ΩÎ°ú ÏßÄ??
        
        Returns:
            PipelineResult: ?åÏù¥?ÑÎùº???§Ìñâ Í≤∞Í≥º
        """
        self.pipeline_state['start_time'] = datetime.now()
        self.logger.info(f"Starting auto pipeline for data source: {data_source}")
        
        stage_results = {}
        error_messages = []
        
        try:
            # Step 1: ?∞Ïù¥??Í∞êÏ?
            self.pipeline_state['current_stage'] = 'detection'
            self.logger.info("Step 1: Detecting new data sources...")
            
            if specific_path:
                detected_files = self._detect_specific_path(specific_path, data_source)
            elif auto_detect:
                detected_files = self._detect_new_data_sources(data_source)
            else:
                self.logger.error("No detection method specified")
                return PipelineResult(
                    success=False,
                    total_files_detected=0,
                    files_processed=0,
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results={},
                    error_messages=["No detection method specified"]
                )
            
            stage_results['detection'] = {
                'success': len(detected_files) > 0,
                'files_detected': sum(len(files) for files in detected_files.values()),
                'data_types': list(detected_files.keys())
            }
            
            if not detected_files:
                self.logger.info("No new files detected")
                return PipelineResult(
                    success=True,
                    total_files_detected=0,
                    files_processed=0,
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results=stage_results,
                    error_messages=[]
                )
            
            # Step 2: Ï¶ùÎ∂Ñ ?ÑÏ≤òÎ¶?
            self.pipeline_state['current_stage'] = 'preprocessing'
            self.logger.info("Step 2: Incremental preprocessing...")
            
            preprocessing_results = self._run_preprocessing_stage(detected_files)
            stage_results['preprocessing'] = preprocessing_results
            
            if not preprocessing_results['success']:
                error_messages.extend(preprocessing_results['errors'])
                return PipelineResult(
                    success=False,
                    total_files_detected=sum(len(files) for files in detected_files.values()),
                    files_processed=0,
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results=stage_results,
                    error_messages=error_messages
                )
            
            # Step 3: ?àÏßà Í≤ÄÏ¶?Î∞?Í∞úÏÑ†
            self.pipeline_state['current_stage'] = 'quality_validation'
            self.logger.info("Step 3: Quality validation and improvement...")
            
            quality_results = self._run_quality_validation_stage(preprocessing_results['processed_files'])
            stage_results['quality_validation'] = quality_results
            
            if not quality_results['success']:
                error_messages.extend(quality_results['errors'])
                # ?àÏßà Í≤ÄÏ¶??§Ìå®??Í≤ΩÍ≥†Î°?Ï≤òÎ¶¨?òÍ≥† Í≥ÑÏÜç ÏßÑÌñâ
                self.logger.warning(f"Quality validation failed: {quality_results['errors']}")
            
            # Step 4: Ï¶ùÎ∂Ñ Î≤°ÌÑ∞ ?ÑÎ≤†??
            self.pipeline_state['current_stage'] = 'vectorization'
            self.logger.info("Step 4: Incremental vector embedding...")
            
            vectorization_results = self._run_vectorization_stage(preprocessing_results['processed_files'])
            stage_results['vectorization'] = vectorization_results
            
            if not vectorization_results['success']:
                error_messages.extend(vectorization_results['errors'])
                return PipelineResult(
                    success=False,
                    total_files_detected=sum(len(files) for files in detected_files.values()),
                    files_processed=preprocessing_results['processed_files_count'],
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results=stage_results,
                    error_messages=error_messages
                )
            
            # Step 5: DB Ï¶ùÎ∂Ñ ?ÑÌè¨??
            self.pipeline_state['current_stage'] = 'import'
            self.logger.info("Step 5: Incremental database import...")
            
            import_results = self._run_import_stage(preprocessing_results['processed_files'])
            stage_results['import'] = import_results
            
            if not import_results['success']:
                error_messages.extend(import_results['errors'])
            
            # Step 6: ÏµúÏ¢Ö ?µÍ≥Ñ ?ùÏÑ±
            self.pipeline_state['current_stage'] = 'finalization'
            self.logger.info("Step 6: Generating final statistics...")
            
            final_stats = self._generate_final_statistics()
            stage_results['finalization'] = final_stats
            
            # ?åÏù¥?ÑÎùº???ÑÎ£å
            self.pipeline_state['end_time'] = datetime.now()
            processing_time = (
                self.pipeline_state['end_time'] - self.pipeline_state['start_time']
            ).total_seconds()
            
            success = len(error_messages) == 0
            
            result = PipelineResult(
                success=success,
                total_files_detected=sum(len(files) for files in detected_files.values()),
                files_processed=preprocessing_results['processed_files_count'],
                vectors_added=vectorization_results['vectors_added'],
                laws_imported=import_results['laws_imported'],
                processing_time=processing_time,
                stage_results=stage_results,
                error_messages=error_messages,
                quality_metrics=quality_results.get('metrics', {}),
                quality_improvements=quality_results.get('improvements_made', 0),
                duplicates_resolved=quality_results.get('duplicates_resolved', 0)
            )
            
            self.logger.info(f"Pipeline completed: {success}")
            self.logger.info(f"Total files detected: {result.total_files_detected}")
            self.logger.info(f"Files processed: {result.files_processed}")
            self.logger.info(f"Vectors added: {result.vectors_added}")
            self.logger.info(f"Laws imported: {result.laws_imported}")
            self.logger.info(f"Processing time: {result.processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline execution error: {e}"
            self.logger.error(error_msg)
            error_messages.append(error_msg)
            
            return PipelineResult(
                success=False,
                total_files_detected=0,
                files_processed=0,
                vectors_added=0,
                laws_imported=0,
                processing_time=0,
                stage_results=stage_results,
                error_messages=error_messages
            )
    
    def run_precedent_pipeline(self, 
                              category: str = "civil",
                              auto_detect: bool = True,
                              specific_path: str = None) -> PipelineResult:
        """
        ?êÎ? ?∞Ïù¥???êÎèô???åÏù¥?ÑÎùº???§Ìñâ
        
        Args:
            category: ?êÎ? Ïπ¥ÌÖåÍ≥†Î¶¨ (civil, criminal, family)
            auto_detect: ?êÎèô Í∞êÏ? ?¨Î?
            specific_path: ?πÏ†ï Í≤ΩÎ°ú ÏßÄ??
        
        Returns:
            PipelineResult: ?åÏù¥?ÑÎùº???§Ìñâ Í≤∞Í≥º
        """
        self.pipeline_state['start_time'] = datetime.now()
        self.logger.info(f"Starting precedent pipeline for category: {category}")
        
        stage_results = {}
        error_messages = []
        
        try:
            # Step 1: ?êÎ? ?∞Ïù¥??Í∞êÏ?
            self.pipeline_state['current_stage'] = 'detection'
            self.logger.info("Step 1: Detecting new precedent data sources...")
            
            data_type = f"precedent_{category}"
            if specific_path:
                detected_files = self._detect_specific_path(specific_path, data_type)
            elif auto_detect:
                detected_files = self._detect_new_data_sources(data_type)
            else:
                self.logger.error("No detection method specified")
                return PipelineResult(
                    success=False,
                    total_files_detected=0,
                    files_processed=0,
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results=stage_results,
                    error_messages=["No detection method specified"]
                )
            
            stage_results['detection'] = {
                'total_files': sum(len(files) for files in detected_files.values()),
                'files_by_type': {k: len(v) for k, v in detected_files.items()}
            }
            
            if not detected_files or sum(len(files) for files in detected_files.values()) == 0:
                self.logger.info("No new precedent files detected")
                return PipelineResult(
                    success=True,
                    total_files_detected=0,
                    files_processed=0,
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results=stage_results,
                    error_messages=[]
                )
            
            # Step 2: ?êÎ? ?ÑÏ≤òÎ¶?
            self.pipeline_state['current_stage'] = 'preprocessing'
            self.logger.info("Step 2: Precedent preprocessing...")
            
            preprocessing_results = self.precedent_preprocessor.process_new_data_only(category)
            stage_results['preprocessing'] = preprocessing_results
            
            if preprocessing_results['failed_to_process'] > 0:
                error_messages.extend(preprocessing_results['errors'])
            
            # Step 3: ?êÎ? Î≤°ÌÑ∞ ?ÑÎ≤†??
            self.pipeline_state['current_stage'] = 'vectorization'
            self.logger.info("Step 3: Precedent vector embedding...")
            
            vectorization_results = self.precedent_vector_builder.build_incremental_embeddings(category)
            stage_results['vectorization'] = vectorization_results
            
            if vectorization_results['failed_embedding_files'] > 0:
                error_messages.extend(vectorization_results['errors'])
            
            # Step 4: ?êÎ? DB Ï¶ùÎ∂Ñ ?ÑÌè¨??
            self.pipeline_state['current_stage'] = 'import'
            self.logger.info("Step 4: Precedent database import...")
            
            import_results = self._run_precedent_import_stage(category)
            stage_results['import'] = import_results
            
            if not import_results['success']:
                error_messages.extend(import_results['errors'])
            
            # Step 5: ÏµúÏ¢Ö ?µÍ≥Ñ ?ùÏÑ±
            self.pipeline_state['current_stage'] = 'finalization'
            self.logger.info("Step 5: Generating final statistics...")
            
            final_stats = self._generate_final_statistics()
            stage_results['finalization'] = final_stats
            
            # ?åÏù¥?ÑÎùº???ÑÎ£å
            self.pipeline_state['end_time'] = datetime.now()
            processing_time = (
                self.pipeline_state['end_time'] - self.pipeline_state['start_time']
            ).total_seconds()
            
            success = len(error_messages) == 0
            
            result = PipelineResult(
                success=success,
                total_files_detected=sum(len(files) for files in detected_files.values()),
                files_processed=preprocessing_results['successfully_processed'],
                vectors_added=vectorization_results['total_chunks_added'],
                laws_imported=import_results['cases_imported'],
                processing_time=processing_time,
                stage_results=stage_results,
                error_messages=error_messages
            )
            
            self.logger.info(f"Precedent pipeline completed successfully: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Precedent pipeline execution error: {e}"
            self.logger.error(error_msg)
            error_messages.append(error_msg)
            
            return PipelineResult(
                success=False,
                total_files_detected=0,
                files_processed=0,
                vectors_added=0,
                laws_imported=0,
                processing_time=0,
                stage_results=stage_results,
                error_messages=error_messages
            )
    
    def _detect_new_data_sources(self, data_source: str) -> Dict[str, List[Path]]:
        """?àÎ°ú???∞Ïù¥???åÏä§ Í∞êÏ?"""
        try:
            base_path = self.config['data_sources'][data_source]['raw_path']
            detected_files = self.data_detector.detect_new_data_sources(base_path, data_source)
            
            self.logger.info(f"Detected {sum(len(files) for files in detected_files.values())} new files")
            return detected_files
            
        except Exception as e:
            self.logger.error(f"Error in data detection: {e}")
            return {}
    
    def _detect_specific_path(self, specific_path: str, data_source: str) -> Dict[str, List[Path]]:
        """?πÏ†ï Í≤ΩÎ°ú?êÏÑú ?∞Ïù¥??Í∞êÏ?"""
        try:
            path_obj = Path(specific_path)
            if not path_obj.exists():
                self.logger.error(f"Specified path does not exist: {specific_path}")
                return {}
            
            # ?πÏ†ï Í≤ΩÎ°ú???åÏùº?§ÏùÑ Í∞êÏ????åÏùºÎ°?Ï≤òÎ¶¨
            files = list(path_obj.glob("*.json"))
            detected_files = {data_source: files}
            
            self.logger.info(f"Detected {len(files)} files in specific path")
            return detected_files
            
        except Exception as e:
            self.logger.error(f"Error detecting specific path: {e}")
            return {}
    
    def _run_preprocessing_stage(self, detected_files: Dict[str, List[Path]]) -> Dict[str, Any]:
        """?ÑÏ≤òÎ¶??®Í≥Ñ ?§Ìñâ"""
        try:
            all_processed_files = []
            total_records = 0
            errors = []
            
            # ?∞Ïù¥???†ÌòïÎ≥ÑÎ°ú Ï≤òÎ¶¨
            for data_type, files in detected_files.items():
                if not files:
                    continue
                
                self.logger.info(f"Preprocessing {len(files)} {data_type} files...")
                
                result = self.preprocessor.process_new_files_only(files, data_type)
                
                if result.success:
                    all_processed_files.extend(result.processed_files)
                    total_records += result.total_records
                    self.logger.info(f"Successfully processed {len(result.processed_files)} {data_type} files")
                else:
                    errors.extend(result.error_messages)
                    self.logger.error(f"Failed to process {data_type} files")
            
            return {
                'success': len(errors) == 0,
                'processed_files': all_processed_files,
                'processed_files_count': len(all_processed_files),
                'total_records': total_records,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Preprocessing stage error: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'processed_files': [],
                'processed_files_count': 0,
                'total_records': 0,
                'errors': [error_msg]
            }
    
    def _run_quality_validation_stage(self, processed_files: List[Path]) -> Dict[str, Any]:
        """?àÏßà Í≤ÄÏ¶?Î∞?Í∞úÏÑ† ?®Í≥Ñ ?§Ìñâ"""
        try:
            if not QUALITY_MODULES_AVAILABLE or not self.quality_validator:
                self.logger.warning("Quality modules not available, skipping quality validation")
                return {
                    'success': True,
                    'metrics': {},
                    'improvements_made': 0,
                    'duplicates_resolved': 0,
                    'errors': []
                }
            
            if not processed_files:
                self.logger.info("No processed files for quality validation")
                return {
                    'success': True,
                    'metrics': {},
                    'improvements_made': 0,
                    'duplicates_resolved': 0,
                    'errors': []
                }
            
            self.logger.info(f"Running quality validation on {len(processed_files)} processed files...")
            
            quality_metrics = {
                'total_files_validated': 0,
                'high_quality_files': 0,
                'medium_quality_files': 0,
                'low_quality_files': 0,
                'average_quality_score': 0.0,
                'files_requiring_improvement': 0
            }
            
            improvements_made = 0
            duplicates_resolved = 0
            errors = []
            
            # Í∞?Ï≤òÎ¶¨???åÏùº???Ä???àÏßà Í≤ÄÏ¶??òÌñâ
            for file_path in processed_files:
                try:
                    # JSON ?åÏùº?êÏÑú Î≤ïÎ•† ?∞Ïù¥??Î°úÎìú
                    with open(file_path, 'r', encoding='utf-8') as f:
                        law_data = json.load(f)
                    
                    # ?àÏßà ?êÏàò Í≥ÑÏÇ∞
                    quality_score = self.quality_validator.calculate_quality_score(law_data)
                    quality_metrics['total_files_validated'] += 1
                    quality_metrics['average_quality_score'] += quality_score
                    
                    # ?àÏßà ?±Í∏â Î∂ÑÎ•ò
                    if quality_score >= 0.8:
                        quality_metrics['high_quality_files'] += 1
                    elif quality_score >= 0.6:
                        quality_metrics['medium_quality_files'] += 1
                    else:
                        quality_metrics['low_quality_files'] += 1
                        quality_metrics['files_requiring_improvement'] += 1
                    
                    # ?àÏßà ?êÏàòÎ•?Î≤ïÎ•† ?∞Ïù¥?∞Ïóê Ï∂îÍ?
                    law_data['quality_score'] = quality_score
                    law_data['quality_validated'] = True
                    law_data['quality_validation_timestamp'] = datetime.now().isoformat()
                    
                    # Í∞úÏÑ†???∞Ïù¥?∞Î? ?åÏùº???Ä??
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(law_data, f, ensure_ascii=False, indent=2)
                    
                    improvements_made += 1
                    
                except Exception as e:
                    error_msg = f"Error validating quality for {file_path}: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            # ?âÍ∑† ?àÏßà ?êÏàò Í≥ÑÏÇ∞
            if quality_metrics['total_files_validated'] > 0:
                quality_metrics['average_quality_score'] /= quality_metrics['total_files_validated']
            
            # Ï§ëÎ≥µ Í≤Ä??Î∞??¥Í≤∞ (Í∞ÑÎã®???åÏùº Í∏∞Î∞ò Í≤Ä??
            if self.data_cleaner:
                try:
                    # ?ºÏùº ?ïÎ¶¨ ?ëÏóÖ ?§Ìñâ (Ï§ëÎ≥µ ?¥Í≤∞ ?¨Ìï®)
                    cleaning_report = self.data_cleaner.run_daily_cleaning()
                    duplicates_resolved = cleaning_report.duplicates_resolved
                    self.logger.info(f"Resolved {duplicates_resolved} duplicates during quality validation")
                except Exception as e:
                    error_msg = f"Error during duplicate resolution: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            self.logger.info(f"Quality validation completed: {quality_metrics['total_files_validated']} files validated, {improvements_made} improvements made")
            
            return {
                'success': len(errors) == 0,
                'metrics': quality_metrics,
                'improvements_made': improvements_made,
                'duplicates_resolved': duplicates_resolved,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Quality validation stage error: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'metrics': {},
                'improvements_made': 0,
                'duplicates_resolved': 0,
                'errors': [error_msg]
            }
    
    def _run_vectorization_stage(self, processed_files: List[Path]) -> Dict[str, Any]:
        """Î≤°ÌÑ∞???®Í≥Ñ ?§Ìñâ"""
        try:
            if not processed_files:
                self.logger.info("No files to vectorize")
                return {
                    'success': True,
                    'vectors_added': 0,
                    'errors': []
                }
            
            # Í∏∞Ï°¥ ?∏Îç±??Î°úÎìú
            existing_index_path = self.config['vectorization']['existing_index_path']
            if not self.vector_builder.load_existing_index(existing_index_path):
                self.logger.error(f"Failed to load existing index from: {existing_index_path}")
                return {
                    'success': False,
                    'vectors_added': 0,
                    'errors': [f"Failed to load existing index from: {existing_index_path}"]
                }
            
            # ?àÎ°ú??Î¨∏ÏÑú Ï∂îÍ?
            self.logger.info(f"Adding {len(processed_files)} processed files to vector index...")
            result = self.vector_builder.add_new_documents(processed_files)
            
            if result.success:
                # ?ÖÎç∞?¥Ìä∏???∏Îç±???Ä??
                output_path = self.config['vectorization']['output_path']
                if self.vector_builder.save_updated_index(output_path):
                    self.logger.info(f"Successfully added {result.new_vectors} vectors")
                    return {
                        'success': True,
                        'vectors_added': result.new_vectors,
                        'errors': []
                    }
                else:
                    return {
                        'success': False,
                        'vectors_added': 0,
                        'errors': ["Failed to save updated index"]
                    }
            else:
                return {
                    'success': False,
                    'vectors_added': 0,
                    'errors': result.error_messages
                }
            
        except Exception as e:
            error_msg = f"Vectorization stage error: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'vectors_added': 0,
                'errors': [error_msg]
            }
    
    def _run_import_stage(self, processed_files: List[Path]) -> Dict[str, Any]:
        """DB ?ÑÌè¨???®Í≥Ñ ?§Ìñâ"""
        try:
            if not processed_files:
                self.logger.info("No files to import")
                return {
                    'success': True,
                    'laws_imported': 0,
                    'errors': []
                }
            
            total_imported = 0
            total_updated = 0
            total_skipped = 0
            errors = []
            
            # ?åÏùºÎ≥ÑÎ°ú Ï¶ùÎ∂Ñ ?ÑÌè¨??
            for file_path in tqdm(processed_files, desc="Importing to database"):
                try:
                    result = self.db_importer.import_file(file_path, incremental=True)
                    
                    if 'error' not in result:
                        total_imported += result.get('imported_laws', 0)
                        total_updated += result.get('updated_laws', 0)
                        total_skipped += result.get('skipped_laws', 0)
                    else:
                        errors.append(result['error'])
                        
                except Exception as e:
                    error_msg = f"Error importing {file_path}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            success = len(errors) == 0
            
            self.logger.info(f"Import completed: {total_imported} imported, "
                           f"{total_updated} updated, {total_skipped} skipped")
            
            return {
                'success': success,
                'laws_imported': total_imported + total_updated,
                'laws_updated': total_updated,
                'laws_skipped': total_skipped,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Import stage error: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'laws_imported': 0,
                'errors': [error_msg]
            }
    
    def _run_precedent_import_stage(self, category: str) -> Dict[str, Any]:
        """?êÎ? DB ?ÑÌè¨???®Í≥Ñ ?§Ìñâ"""
        try:
            # ?ÑÏ≤òÎ¶¨Îêú ?êÎ? ?åÏùº?§Ïù¥ ?àÎäî ?îÎ†â?†Î¶¨ Ï∞æÍ∏∞
            processed_dir = Path(f"data/processed/assembly/precedent/{category}")
            
            if not processed_dir.exists():
                self.logger.info(f"No processed precedent directory found: {processed_dir}")
                return {
                    'success': True,
                    'cases_imported': 0,
                    'errors': []
                }
            
            # ?†ÏßúÎ≥??îÎ†â?†Î¶¨?êÏÑú ?åÏùº??Ï∞æÍ∏∞
            total_imported = 0
            total_updated = 0
            total_skipped = 0
            errors = []
            
            for date_dir in processed_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                
                # ?¥Îãπ ?†Ïßú ?îÎ†â?†Î¶¨??Î™®Îì† JSON ?åÏùº ?ÑÌè¨??
                json_files = list(date_dir.glob("ml_enhanced_*.json"))
                
                for file_path in json_files:
                    try:
                        result = self.precedent_db_importer.import_file(file_path, incremental=True)
                        
                        if 'error' not in result:
                            total_imported += result.get('imported_cases', 0)
                            total_updated += result.get('updated_cases', 0)
                            total_skipped += result.get('skipped_cases', 0)
                        else:
                            errors.append(result['error'])
                            
                    except Exception as e:
                        error_msg = f"Error importing {file_path}: {e}"
                        errors.append(error_msg)
                        self.logger.error(error_msg)
            
            success = len(errors) == 0
            
            self.logger.info(f"Precedent import completed: {total_imported} imported, "
                           f"{total_updated} updated, {total_skipped} skipped")
            
            return {
                'success': success,
                'cases_imported': total_imported + total_updated,
                'cases_updated': total_updated,
                'cases_skipped': total_skipped,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Precedent import stage error: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'cases_imported': 0,
                'errors': [error_msg]
            }
    
    def _generate_final_statistics(self) -> Dict[str, Any]:
        """ÏµúÏ¢Ö ?µÍ≥Ñ ?ùÏÑ±"""
        try:
            # ?∞Ïù¥?∞Î≤†?¥Ïä§ ?µÍ≥Ñ
            db_stats = self.db_manager.get_processing_statistics()
            
            # Ï≤òÎ¶¨???åÏùº ?µÍ≥Ñ
            processed_files_stats = self.db_manager.get_processed_files_by_type('law_only')
            
            return {
                'success': True,
                'database_statistics': db_stats,
                'processed_files_count': len(processed_files_stats),
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating final statistics: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Í∏∞Î≥∏ ?§Ï†ï Î∞òÌôò"""
        return {
            'data_sources': {
                'law_only': {
                    'enabled': True,
                    'priority': 1,
                    'raw_path': 'data/raw/assembly/law_only',
                    'processed_path': 'data/processed/assembly/law_only'
                },
                'precedent_civil': {
                    'enabled': True,
                    'priority': 2,
                    'raw_path': 'data/raw/assembly/precedent',
                    'processed_path': 'data/processed/assembly/precedent/civil'
                },
                'precedent_criminal': {
                    'enabled': True,
                    'priority': 3,
                    'raw_path': 'data/raw/assembly/precedent',
                    'processed_path': 'data/processed/assembly/precedent/criminal'
                },
                'precedent_family': {
                    'enabled': True,
                    'priority': 4,
                    'raw_path': 'data/raw/assembly/precedent',
                    'processed_path': 'data/processed/assembly/precedent/family'
                },
                'precedent_tax': {
                    'enabled': True,
                    'priority': 5,
                    'raw_path': 'data/raw/assembly/precedent',
                    'processed_path': 'data/processed/assembly/precedent/tax'
                }
            },
            'preprocessing': {
                'batch_size': 100,
                'enable_term_normalization': True,
                'enable_ml_enhancement': True
            },
            'vectorization': {
                'model_name': 'jhgan/ko-sroberta-multitask',
                'dimension': 768,
                'batch_size': 20,
                'chunk_size': 200,
                'index_type': 'flat',
                'existing_index_path': 'data/embeddings/ml_enhanced_ko_sroberta',
                'output_path': 'data/embeddings/ml_enhanced_ko_sroberta',
                'precedent_index_path': 'data/embeddings/ml_enhanced_ko_sroberta_precedents'
            },
            'incremental': {
                'enabled': True,
                'check_file_hash': True,
                'skip_duplicates': True
            },
            'quality': {
                'enabled': True,
                'validation_threshold': 0.7,
                'enable_duplicate_detection': True,
                'enable_quality_improvement': True,
                'quality_thresholds': {
                    'excellent': 0.9,
                    'good': 0.8,
                    'fair': 0.6,
                    'poor': 0.4
                }
            },
            'quality_monitor': {
                'enabled': True,
                'check_interval_seconds': 300,
                'alert_thresholds': {
                    'overall_quality_min': 0.8,
                    'duplicate_max_percentage': 5.0
                }
            }
        }
    
    def save_pipeline_report(self, result: PipelineResult, output_path: str = None):
        """?åÏù¥?ÑÎùº???§Ìñâ Î¶¨Ìè¨???Ä??""
        try:
            if not output_path:
                output_path = f"reports/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report_path = Path(output_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            # datetime Í∞ùÏ≤¥Î•?Î¨∏Ïûê?¥Î°ú Î≥Ä?òÌïò???®Ïàò
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            report_data = {
                'pipeline_result': {
                    'success': result.success,
                    'total_files_detected': result.total_files_detected,
                    'files_processed': result.files_processed,
                    'vectors_added': result.vectors_added,
                    'laws_imported': result.laws_imported,
                    'processing_time': result.processing_time,
                    'error_messages': result.error_messages,
                    'quality_metrics': result.quality_metrics,
                    'quality_improvements': result.quality_improvements,
                    'duplicates_resolved': result.duplicates_resolved
                },
                'stage_results': convert_datetime(result.stage_results),
                'pipeline_state': convert_datetime(self.pipeline_state),
                'config': self.config,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Pipeline report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving pipeline report: {e}")


def main():
    """Î©îÏù∏ ?®Ïàò"""
    parser = argparse.ArgumentParser(description='?êÎèô???åÏù¥?ÑÎùº???§Ï??§Ìä∏?àÏù¥??)
    parser.add_argument('--data-source', default='law_only',
                       choices=['law_only', 'precedents', 'constitutional', 'precedent_civil', 'precedent_criminal', 'precedent_family', 'precedent_tax', 'precedent_administrative', 'precedent_patent'],
                       help='?∞Ïù¥???åÏä§ ?†Ìòï')
    parser.add_argument('--category', default='civil',
                       choices=['civil', 'criminal', 'family', 'tax', 'administrative', 'patent'],
                       help='?êÎ? Ïπ¥ÌÖåÍ≥†Î¶¨ (precedent ?∞Ïù¥???åÏä§ ?¨Ïö© ??')
    parser.add_argument('--auto-detect', action='store_true',
                       help='?êÎèô ?∞Ïù¥??Í∞êÏ? ?úÏÑ±??)
    parser.add_argument('--data-path', help='?πÏ†ï ?∞Ïù¥??Í≤ΩÎ°ú ÏßÄ??)
    parser.add_argument('--config', help='?§Ï†ï ?åÏùº Í≤ΩÎ°ú')
    parser.add_argument('--checkpoint-dir', default='data/checkpoints',
                       help='Ï≤¥ÌÅ¨?¨Ïù∏???îÎ†â?†Î¶¨')
    parser.add_argument('--db-path', default='data/lawfirm.db',
                       help='?∞Ïù¥?∞Î≤†?¥Ïä§ Í≤ΩÎ°ú')
    parser.add_argument('--output-report', help='Î¶¨Ìè¨??Ï∂úÎ†• Í≤ΩÎ°ú')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='?ÅÏÑ∏ Î°úÍ∑∏ Ï∂úÎ†•')
    
    args = parser.parse_args()
    
    # Î°úÍπÖ ?§Ï†ï
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # ?§Ï†ï Î°úÎìú
        config = None
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # ?åÏù¥?ÑÎùº???§Ï??§Ìä∏?àÏù¥??Ï¥àÍ∏∞??
        orchestrator = AutoPipelineOrchestrator(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            db_path=args.db_path
        )
        
        # ?åÏù¥?ÑÎùº???§Ìñâ
        if args.data_source.startswith('precedent_'):
            # ?êÎ? ?åÏù¥?ÑÎùº???§Ìñâ
            category = args.data_source.split('_')[1]  # precedent_civil -> civil
            result = orchestrator.run_precedent_pipeline(
                category=category,
                auto_detect=args.auto_detect,
                specific_path=args.data_path
            )
        else:
            # Î≤ïÎ•† ?åÏù¥?ÑÎùº???§Ìñâ
            result = orchestrator.run_auto_pipeline(
                data_source=args.data_source,
                auto_detect=args.auto_detect,
                specific_path=args.data_path
            )
        
        # Î¶¨Ìè¨???Ä??
        orchestrator.save_pipeline_report(result, args.output_report)
        
        # Í≤∞Í≥º Ï∂úÎ†•
        print("\n" + "="*60)
        print("AUTO PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Success: {result.success}")
        print(f"Total files detected: {result.total_files_detected}")
        print(f"Files processed: {result.files_processed}")
        print(f"Vectors added: {result.vectors_added}")
        print(f"Laws imported: {result.laws_imported}")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        
        if result.error_messages:
            print("\nErrors:")
            for error in result.error_messages:
                print(f"  - {error}")
        
        print("="*60)
        
        return result.success
        
    except Exception as e:
        logging.error(f"Error in auto pipeline: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
