import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import { storage } from "./storage";
import { insertMeasurementSchema } from "@shared/schema";

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
        path.join(process.cwd(), 'server/python/pd_processor_robust.py'),
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
