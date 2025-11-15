-- Migration: Add reference_statutes field to cases, decisions, and interpretations tables
-- This allows pre-extracted legal references to be stored at data ingestion time

-- Add reference_statutes column to cases table
ALTER TABLE cases ADD COLUMN reference_statutes TEXT;

-- Add reference_statutes column to decisions table
ALTER TABLE decisions ADD COLUMN reference_statutes TEXT;

-- Add reference_statutes column to interpretations table
ALTER TABLE interpretations ADD COLUMN reference_statutes TEXT;

-- Create indexes for reference_statutes (for faster lookups)
CREATE INDEX IF NOT EXISTS idx_cases_reference_statutes 
ON cases(reference_statutes) WHERE reference_statutes IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_decisions_reference_statutes 
ON decisions(reference_statutes) WHERE reference_statutes IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_interpretations_reference_statutes 
ON interpretations(reference_statutes) WHERE reference_statutes IS NOT NULL;

