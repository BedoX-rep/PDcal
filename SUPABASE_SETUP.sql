-- Execute these SQL commands in your Supabase SQL Editor
-- This will create the necessary tables for measurement history and image storage

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create measurements table
CREATE TABLE IF NOT EXISTS measurements (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    pd_value DECIMAL(5,2) NOT NULL,
    left_pupil_x INTEGER NOT NULL,
    left_pupil_y INTEGER NOT NULL,
    right_pupil_x INTEGER NOT NULL,
    right_pupil_y INTEGER NOT NULL,
    nose_bridge_x INTEGER,
    nose_bridge_y INTEGER,
    left_monocular_pd DECIMAL(5,2),
    right_monocular_pd DECIMAL(5,2),
    pixel_distance DECIMAL(8,2) NOT NULL,
    scale_factor DECIMAL(8,4) NOT NULL,
    original_image_url TEXT,
    processed_image_url TEXT,
    left_ocular_height DECIMAL(5,2),
    right_ocular_height DECIMAL(5,2),
    ocular_confidence DECIMAL(3,2),
    analysis_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create RLS (Row Level Security) policies
ALTER TABLE measurements ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own measurements
CREATE POLICY "Users can view own measurements" ON measurements
    FOR SELECT USING (auth.uid() = user_id);

-- Policy: Users can insert their own measurements
CREATE POLICY "Users can insert own measurements" ON measurements
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy: Users can update their own measurements
CREATE POLICY "Users can update own measurements" ON measurements
    FOR UPDATE USING (auth.uid() = user_id);

-- Policy: Users can delete their own measurements
CREATE POLICY "Users can delete own measurements" ON measurements
    FOR DELETE USING (auth.uid() = user_id);

-- Create storage bucket for images (if not exists)
INSERT INTO storage.buckets (id, name, public) 
VALUES ('measurement-images', 'measurement-images', false)
ON CONFLICT (id) DO NOTHING;

-- Create storage policies for the bucket
CREATE POLICY "Users can upload their own images" ON storage.objects
    FOR INSERT WITH CHECK (
        bucket_id = 'measurement-images' 
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

CREATE POLICY "Users can view their own images" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'measurement-images' 
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

CREATE POLICY "Users can delete their own images" ON storage.objects
    FOR DELETE USING (
        bucket_id = 'measurement-images' 
        AND auth.uid()::text = (storage.foldername(name))[1]
    );