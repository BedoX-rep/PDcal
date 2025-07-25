import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import { supabaseAdmin, getUserFromRequest } from "./supabase";
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
      if (!req.file) {
        return res.status(400).json({ error: "No image file provided" });
      }

      const imagePath = req.file.path;
      const originalFilename = req.file.originalname;

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
            const measurement = await storage.createMeasurement({
              originalFilename,
              processedImagePath: null,
              pdValue: null,
              leftPupilX: null,
              leftPupilY: null,
              rightPupilX: null,
              rightPupilY: null,
              noseBridgeX: null,
              noseBridgeY: null,
              leftMonocularPd: null,
              rightMonocularPd: null,
              pixelDistance: null,
              scaleFactor: null,
              apriltagDetected: false,
              pupilsDetected: false,
              errorMessage: "Failed to process image: Could not detect pupils or AprilTag"
            });
            return res.status(400).json({ error: measurement.errorMessage, measurementId: measurement.id });
          }

          // Parse Python output
          const result = JSON.parse(pythonOutput);

          if (!result.success) {
            const measurement = await storage.createMeasurement({
              originalFilename,
              processedImagePath: null,
              pdValue: null,
              leftPupilX: null,
              leftPupilY: null,
              rightPupilX: null,
              rightPupilY: null,
              noseBridgeX: null,
              noseBridgeY: null,
              leftMonocularPd: null,
              rightMonocularPd: null,
              pixelDistance: null,
              scaleFactor: null,
              apriltagDetected: result.apriltag_detected || false,
              pupilsDetected: result.pupils_detected || false,
              errorMessage: result.error
            });
            return res.status(400).json({ error: result.error, measurementId: measurement.id });
          }

          // Create successful measurement record
          const measurement = await storage.createMeasurement({
            originalFilename,
            processedImagePath: result.processed_image_path,
            pdValue: result.pd_value,
            leftPupilX: result.left_pupil.x,
            leftPupilY: result.left_pupil.y,
            rightPupilX: result.right_pupil.x,
            rightPupilY: result.right_pupil.y,
            noseBridgeX: result.nose_bridge.x,
            noseBridgeY: result.nose_bridge.y,
            leftMonocularPd: result.left_monocular_pd,
            rightMonocularPd: result.right_monocular_pd,
            pixelDistance: result.pixel_distance,
            scaleFactor: result.scale_factor,
            apriltagDetected: true,
            pupilsDetected: true,
            errorMessage: null
          });

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

  // Get measurement by ID
  app.get("/api/measurements/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const measurement = await storage.getMeasurement(id);
      
      if (!measurement) {
        return res.status(404).json({ error: "Measurement not found" });
      }

      res.json(measurement);
    } catch (error) {
      console.error('Get measurement error:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Get all measurements
  app.get("/api/measurements", async (req, res) => {
    try {
      const measurements = await storage.getAllMeasurements();
      res.json(measurements);
    } catch (error) {
      console.error('Get measurements error:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Analyze ocular height using Gemini
  app.post("/api/measurements/:id/ocular-height", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const measurement = await storage.getMeasurement(id);
      
      if (!measurement) {
        return res.status(404).json({ error: "Measurement not found" });
      }

      if (!measurement.processedImagePath || !measurement.pdValue) {
        return res.status(400).json({ error: "Measurement must have processed image and PD value" });
      }

      const imagePath = path.join(process.cwd(), 'server/processed_images', measurement.processedImagePath);
      
      if (!fs.existsSync(imagePath)) {
        return res.status(404).json({ error: "Processed image not found" });
      }

      // Analyze ocular height with Gemini
      const ocularAnalysis = await analyzeOcularHeight(imagePath, measurement.pdValue);

      // Update measurement with ocular height data
      const updatedMeasurement = await storage.updateMeasurement(id, {
        leftOcularHeight: ocularAnalysis.leftOcularHeight,
        rightOcularHeight: ocularAnalysis.rightOcularHeight,
        ocularHeightAnalyzed: true,
      });

      res.json({
        success: true,
        measurement: updatedMeasurement,
        ocularAnalysis,
      });

    } catch (error: any) {
      console.error('Ocular height analysis error:', error);
      res.status(500).json({ error: error.message || "Failed to analyze ocular height" });
    }
  });

  // Manual ocular height calculation
  app.post("/api/measurements/:id/manual-ocular-height", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const { leftFrameBottomY, rightFrameBottomY, zoomLevel, imageWidth, imageHeight } = req.body;
      
      if (!leftFrameBottomY || !rightFrameBottomY || !zoomLevel || !imageWidth || !imageHeight) {
        return res.status(400).json({ error: "leftFrameBottomY, rightFrameBottomY, zoomLevel, imageWidth, and imageHeight are required" });
      }

      const measurement = await storage.getMeasurement(id);
      
      if (!measurement) {
        return res.status(404).json({ error: "Measurement not found" });
      }

      if (!measurement.leftPupilX || !measurement.leftPupilY || !measurement.rightPupilX || !measurement.rightPupilY || !measurement.scaleFactor) {
        return res.status(400).json({ error: "Measurement must have pupil coordinates and scale factor" });
      }

      // Calculate ocular heights based on manual line placement for each eye separately
      // Distance from pupil center to respective frame bottom line in pixels
      const leftOcularPixels = Math.abs(leftFrameBottomY - measurement.leftPupilY);
      const rightOcularPixels = Math.abs(rightFrameBottomY - measurement.rightPupilY);
      
      // Convert to millimeters using the scale factor from the measurement
      const leftOcularHeight = leftOcularPixels * measurement.scaleFactor;
      const rightOcularHeight = rightOcularPixels * measurement.scaleFactor;

      // Update measurement with manual ocular height data
      const updatedMeasurement = await storage.updateMeasurement(id, {
        leftOcularHeight,
        rightOcularHeight,
        ocularHeightAnalyzed: true,
      });

      const ocularAnalysis = {
        leftOcularHeight,
        rightOcularHeight,
        confidence: 1.0, // Manual placement is considered 100% confident
        analysisNotes: `Manual measurement with separate frame lines: Left eye frame at Y=${leftFrameBottomY.toFixed(1)}px (distance: ${leftOcularPixels.toFixed(1)}px = ${leftOcularHeight.toFixed(1)}mm), Right eye frame at Y=${rightFrameBottomY.toFixed(1)}px (distance: ${rightOcularPixels.toFixed(1)}px = ${rightOcularHeight.toFixed(1)}mm). Scale factor: ${measurement.scaleFactor.toFixed(4)} mm/pixel.`
      };

      res.json({
        success: true,
        measurement: updatedMeasurement,
        ocularAnalysis,
      });

    } catch (error: any) {
      console.error('Manual ocular height calculation error:', error);
      res.status(500).json({ error: error.message || "Failed to calculate manual ocular height" });
    }
  });

  // Test endpoint for direct ocular height analysis
  app.post("/api/test-ocular-height", async (req, res) => {
    try {
      const { imagePath, pdValue } = req.body;
      
      if (!imagePath || !pdValue) {
        return res.status(400).json({ error: "imagePath and pdValue are required" });
      }
      
      if (!fs.existsSync(imagePath)) {
        return res.status(404).json({ error: "Image file not found" });
      }

      // Analyze ocular height with Gemini
      const ocularAnalysis = await analyzeOcularHeight(imagePath, pdValue);

      res.json({
        success: true,
        ocularAnalysis,
      });

    } catch (error: any) {
      console.error('Direct ocular height analysis error:', error);
      res.status(500).json({ error: error.message || "Failed to analyze ocular height" });
    }
  });

  // Serve processed images
  app.get("/api/images/:filename", (req, res) => {
    const filename = req.params.filename;
    const imagePath = path.join(process.cwd(), 'server/processed_images', filename);
    
    if (fs.existsSync(imagePath)) {
      res.sendFile(imagePath);
    } else {
      res.status(404).json({ error: "Image not found" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
