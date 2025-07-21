import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import { supabaseAdmin, getUserFromRequest, createAuthenticatedClient } from "./supabase";
import { analyzeOcularHeight } from "./gemini";

// Configure multer for file uploads
const upload = multer({ 
  dest: 'server/uploads/',
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only JPG, PNG, and WEBP are allowed.'));
    }
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Upload and process image endpoint
  app.post("/api/measurements", upload.single('image'), async (req, res) => {
    try {
      // Get user from token and create authenticated client
      const authHeader = req.headers.authorization;
      if (!authHeader) {
        return res.status(401).json({ error: "Unauthorized - No auth header" });
      }
      
      const token = authHeader.replace('Bearer ', '');
      const user = await getUserFromRequest(req);
      if (!user) {
        return res.status(401).json({ error: "Unauthorized - Invalid token" });
      }
      
      const supabase = createAuthenticatedClient(token);

      if (!req.file) {
        return res.status(400).json({ error: "No image file provided" });
      }

      const imagePath = req.file.path;

      // Call Python script for image processing
      const pythonProcess = spawn('python3', [
        path.join(process.cwd(), 'server/python/pd_processor_final.py'),
        imagePath
      ]);

      let pythonOutput = '';
      let pythonError = '';

      pythonProcess.stdout.on('data', (data) => {
        pythonOutput += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
      });

      pythonProcess.on('close', async (code) => {
        try {
          // Clean up uploaded file
          if (fs.existsSync(imagePath)) {
            fs.unlinkSync(imagePath);
          }

          if (code !== 0) {
            console.error('Python script error:', pythonError);
            return res.status(400).json({ error: "Failed to process image: Could not detect pupils or AprilTag" });
          }

          // Parse Python output
          const result = JSON.parse(pythonOutput);

          if (!result.success) {
            return res.status(400).json({ error: result.error || "Processing failed" });
          }

          // Save to Supabase with user authentication context
          const { data: measurement, error } = await supabase
            .from('measurements')
            .insert({
              user_id: user.id,
              pd_value: parseFloat(result.pd_value),
              left_pupil_x: result.left_pupil.x,
              left_pupil_y: result.left_pupil.y,
              right_pupil_x: result.right_pupil.x,
              right_pupil_y: result.right_pupil.y,
              nose_bridge_x: result.nose_bridge?.x || null,
              nose_bridge_y: result.nose_bridge?.y || null,
              left_monocular_pd: result.left_monocular_pd ? parseFloat(result.left_monocular_pd) : null,
              right_monocular_pd: result.right_monocular_pd ? parseFloat(result.right_monocular_pd) : null,
              pixel_distance: parseFloat(result.pixel_distance),
              scale_factor: parseFloat(result.scale_factor),
              processed_image_url: result.processed_image_path,
            })
            .select()
            .single();

          if (error) {
            console.error('Supabase error:', error);
            return res.status(500).json({ error: "Failed to save measurement" });
          }

          res.json({
            success: true,
            measurement,
            result
          });

        } catch (error) {
          console.error('Error processing Python output:', error);
          res.status(500).json({ error: "Internal server error during image processing" });
        }
      });

    } catch (error) {
      console.error('Upload error:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Get all measurements for user
  app.get("/api/measurements", async (req, res) => {
    try {
      const user = await getUserFromRequest(req);
      if (!user) {
        return res.status(401).json({ error: "Unauthorized" });
      }

      const { data: measurements, error } = await supabaseAdmin
        .from('measurements')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Supabase error:', error);
        return res.status(500).json({ error: "Failed to fetch measurements" });
      }

      res.json(measurements || []);
    } catch (error) {
      console.error('Get measurements error:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Get measurement by ID
  app.get("/api/measurements/:id", async (req, res) => {
    try {
      const user = await getUserFromRequest(req);
      if (!user) {
        return res.status(401).json({ error: "Unauthorized" });
      }

      const id = parseInt(req.params.id);
      const { data: measurement, error } = await supabaseAdmin
        .from('measurements')
        .select('*')
        .eq('id', id)
        .eq('user_id', user.id)
        .single();

      if (error || !measurement) {
        return res.status(404).json({ error: "Measurement not found" });
      }

      res.json(measurement);
    } catch (error) {
      console.error('Get measurement error:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Update measurement 
  app.patch("/api/measurements/:id", async (req, res) => {
    try {
      const user = await getUserFromRequest(req);
      if (!user) {
        return res.status(401).json({ error: "Unauthorized" });
      }

      const id = parseInt(req.params.id);
      const { measurementName } = req.body;

      const updateData: any = {};
      if (measurementName !== undefined) {
        updateData.measurement_name = measurementName;
      }

      if (Object.keys(updateData).length === 0) {
        return res.status(400).json({ error: "No valid fields to update" });
      }

      const { data: measurement, error } = await supabaseAdmin
        .from('measurements')
        .update(updateData)
        .eq('id', id)
        .eq('user_id', user.id)
        .select()
        .single();

      if (error || !measurement) {
        console.error('Supabase error:', error);
        return res.status(404).json({ error: "Measurement not found or update failed" });
      }

      res.json(measurement);
    } catch (error) {
      console.error('Update measurement error:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Delete measurement
  app.delete("/api/measurements/:id", async (req, res) => {
    try {
      const user = await getUserFromRequest(req);
      if (!user) {
        return res.status(401).json({ error: "Unauthorized" });
      }

      const id = parseInt(req.params.id);
      const { error } = await supabaseAdmin
        .from('measurements')
        .delete()
        .eq('id', id)
        .eq('user_id', user.id);

      if (error) {
        console.error('Supabase error:', error);
        return res.status(500).json({ error: "Failed to delete measurement" });
      }

      res.json({ success: true });
    } catch (error) {
      console.error('Delete measurement error:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Analyze ocular height using Gemini
  app.post("/api/measurements/:id/ocular-height", async (req, res) => {
    try {
      const user = await getUserFromRequest(req);
      if (!user) {
        return res.status(401).json({ error: "Unauthorized" });
      }

      const id = parseInt(req.params.id);
      
      const { data: measurement, error } = await supabaseAdmin
        .from('measurements')
        .select('*')
        .eq('id', id)
        .eq('user_id', user.id)
        .single();

      if (error || !measurement) {
        return res.status(404).json({ error: "Measurement not found" });
      }

      if (!measurement.processed_image_url) {
        return res.status(400).json({ error: "No processed image available" });
      }

      const imagePath = path.join(process.cwd(), measurement.processed_image_url);
      
      if (!fs.existsSync(imagePath)) {
        return res.status(404).json({ error: "Processed image file not found" });
      }

      const analysis = await analyzeOcularHeight(imagePath);

      // Update measurement with ocular height data
      const { data: updatedMeasurement, error: updateError } = await supabaseAdmin
        .from('measurements')
        .update({
          left_ocular_height: analysis.leftOcularHeight.toString(),
          right_ocular_height: analysis.rightOcularHeight.toString(),
          ocular_confidence: analysis.confidence.toString(),
          analysis_notes: analysis.analysisNotes,
          updated_at: new Date().toISOString()
        })
        .eq('id', id)
        .eq('user_id', user.id)
        .select()
        .single();

      if (updateError) {
        console.error('Update error:', updateError);
        return res.status(500).json({ error: "Failed to update measurement" });
      }

      res.json({
        success: true,
        measurement: updatedMeasurement,
        ocularAnalysis: analysis
      });

    } catch (error) {
      console.error('Ocular height analysis error:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Manual ocular height calculation
  app.post("/api/measurements/:id/manual-ocular-height", async (req, res) => {
    try {
      const user = await getUserFromRequest(req);
      if (!user) {
        return res.status(401).json({ error: "Unauthorized" });
      }

      const id = parseInt(req.params.id);
      const { leftFrameBottomY, rightFrameBottomY, zoomLevel, imageWidth, imageHeight } = req.body;

      const { data: measurement, error } = await supabaseAdmin
        .from('measurements')
        .select('*')
        .eq('id', id)
        .eq('user_id', user.id)
        .single();

      if (error || !measurement) {
        return res.status(404).json({ error: "Measurement not found" });
      }

      // Calculate ocular heights based on manual line placement
      const leftPupilY = measurement.left_pupil_y;
      const rightPupilY = measurement.right_pupil_y;
      const scaleFactor = parseFloat(measurement.scale_factor);

      // Adjust coordinates based on zoom level
      const adjustedLeftFrameY = leftFrameBottomY / zoomLevel;
      const adjustedRightFrameY = rightFrameBottomY / zoomLevel;

      // Calculate pixel distances
      const leftPixelDistance = Math.abs(leftPupilY - adjustedLeftFrameY);
      const rightPixelDistance = Math.abs(rightPupilY - adjustedRightFrameY);

      // Convert to millimeters using scale factor
      const leftOcularHeight = leftPixelDistance * scaleFactor;
      const rightOcularHeight = rightPixelDistance * scaleFactor;

      const analysis = {
        leftOcularHeight: Math.round(leftOcularHeight * 10) / 10,
        rightOcularHeight: Math.round(rightOcularHeight * 10) / 10,
        confidence: 0.95,
        analysisNotes: "Manual measurement using frame bottom line placement"
      };

      // Update measurement
      const { data: updatedMeasurement, error: updateError } = await supabaseAdmin
        .from('measurements')
        .update({
          left_ocular_height: analysis.leftOcularHeight.toString(),
          right_ocular_height: analysis.rightOcularHeight.toString(),
          ocular_confidence: analysis.confidence.toString(),
          analysis_notes: analysis.analysisNotes,
          updated_at: new Date().toISOString()
        })
        .eq('id', id)
        .eq('user_id', user.id)
        .select()
        .single();

      if (updateError) {
        console.error('Update error:', updateError);
        return res.status(500).json({ error: "Failed to update measurement" });
      }

      res.json({
        success: true,
        measurement: updatedMeasurement,
        ocularAnalysis: analysis
      });

    } catch (error) {
      console.error('Manual ocular height error:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Serve processed images
  app.get("/api/images/:filename", (req, res) => {
    const filename = req.params.filename;
    const imagePath = path.join(process.cwd(), 'server/processed_images', filename);
    
    if (fs.existsSync(imagePath)) {
      res.sendFile(path.resolve(imagePath));
    } else {
      res.status(404).json({ error: "Image not found" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}