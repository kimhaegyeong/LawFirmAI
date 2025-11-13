-- Migration: Add Open Law API fields for URL generation
-- Version: 2
-- Description: Add law_id, mst, proclamation_number to statutes table
--              Add precedent_serial_number to cases table
--              Add decision_serial_number to decisions table
--              Add interpretation_serial_number to interpretations table

-- Add Open Law API fields to statutes table
ALTER TABLE statutes ADD COLUMN law_id TEXT;
ALTER TABLE statutes ADD COLUMN mst TEXT;
ALTER TABLE statutes ADD COLUMN proclamation_number TEXT;

-- Add Open Law API fields to cases table
ALTER TABLE cases ADD COLUMN precedent_serial_number TEXT;

-- Add Open Law API fields to decisions table
ALTER TABLE decisions ADD COLUMN decision_serial_number TEXT;

-- Add Open Law API fields to interpretations table
ALTER TABLE interpretations ADD COLUMN interpretation_serial_number TEXT;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_statutes_law_id ON statutes(law_id);
CREATE INDEX IF NOT EXISTS idx_statutes_mst ON statutes(mst);
CREATE INDEX IF NOT EXISTS idx_cases_precedent_serial_number ON cases(precedent_serial_number);
CREATE INDEX IF NOT EXISTS idx_decisions_decision_serial_number ON decisions(decision_serial_number);
CREATE INDEX IF NOT EXISTS idx_interpretations_interpretation_serial_number ON interpretations(interpretation_serial_number);

-- Update user_version
PRAGMA user_version = 2;

