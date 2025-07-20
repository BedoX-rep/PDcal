import * as fs from "fs";
import { GoogleGenAI } from "@google/genai";

// Initialize Gemini with API key from environment
const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY || "" });

export interface OcularHeightResult {
  leftOcularHeight: number;
  rightOcularHeight: number;
  confidence: number;
  analysisNotes: string;
}

// Validation function for ocular height measurements
function validateOcularHeight(result: OcularHeightResult): boolean {
  const { leftOcularHeight, rightOcularHeight, confidence } = result;
  
  // Check if values are within reasonable range (15-40mm for typical adults)
  if (leftOcularHeight < 15 || leftOcularHeight > 40 || 
      rightOcularHeight < 15 || rightOcularHeight > 40) {
    return false;
  }
  
  // Check if confidence is reasonable
  if (confidence < 0.7) {
    return false;
  }
  
  // Check if left and right values are not too different (max 5mm difference)
  const difference = Math.abs(leftOcularHeight - rightOcularHeight);
  if (difference > 5) {
    return false;
  }
  
  return true;
}

export async function analyzeOcularHeight(imagePath: string, pdValue: number): Promise<OcularHeightResult> {
  const maxRetries = 3;
  let lastError: Error | null = null;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const imageBytes = fs.readFileSync(imagePath);

      const prompt = `OPTICAL MEASUREMENT SYSTEM - OCULAR HEIGHT ANALYSIS

INPUT SPECIFICATIONS:
- Image: Processed eye measurement with green pupil markers and AprilTag
- Reference PD: ${pdValue}mm (verified measurement)
- Task: Calculate ocular height (pupil center to frame bottom edge)

MEASUREMENT PROTOCOL (EXECUTE EXACTLY):

STEP 1 - LOCATE MARKERS:
Find the two green circular markers indicating pupil centers.
Record pixel coordinates of each marker center.

STEP 2 - CALCULATE SCALE:
- Measure pixel distance between the two green pupil markers
- Apply scale: ${pdValue}mm ÷ pixel_distance = mm_per_pixel

STEP 3 - MEASURE OCULAR HEIGHT:
For each eye:
- Start at green pupil marker center
- Measure vertically downward to lowest frame edge
- Convert pixels to mm using calculated scale

STEP 4 - QUALITY CHECK:
- Left and right values should be similar (±3mm)
- Typical range: 15-35mm for adult eyeglasses
- Confidence based on marker clarity and frame visibility

CRITICAL REQUIREMENTS:
- Use IDENTICAL scale calculation for both eyes
- Measure from exact CENTER of green markers
- Measure to LOWEST visible frame point below each pupil
- Round to 1 decimal place

Return precise JSON:
{
  "leftOcularHeight": exact_measurement_in_mm,
  "rightOcularHeight": exact_measurement_in_mm,
  "confidence": quality_score_0_to_1,
  "analysisNotes": "Pixel measurements and scale calculation details"
}`;

      const contents = [
        {
          inlineData: {
            data: imageBytes.toString("base64"),
            mimeType: "image/jpeg",
          },
        },
        prompt,
      ];

      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        config: {
          temperature: 0.0, // Maximum determinism
          topP: 0.1,        // Reduce randomness
          responseMimeType: "application/json",
          responseSchema: {
            type: "object",
            properties: {
              leftOcularHeight: { type: "number" },
              rightOcularHeight: { type: "number" },
              confidence: { type: "number" },
              analysisNotes: { type: "string" },
            },
            required: ["leftOcularHeight", "rightOcularHeight", "confidence", "analysisNotes"],
          },
        },
        contents: contents,
      });

      const rawJson = response.text;
      console.log(`Gemini ocular height analysis (attempt ${attempt}): ${rawJson}`);

      if (rawJson) {
        const data: OcularHeightResult = JSON.parse(rawJson);
        
        // Validate the result
        if (validateOcularHeight(data)) {
          console.log(`Ocular height analysis successful on attempt ${attempt}`);
          return data;
        } else {
          console.log(`Validation failed on attempt ${attempt}, retrying...`);
          lastError = new Error(`Invalid measurement ranges or inconsistent values on attempt ${attempt}`);
          continue;
        }
      } else {
        lastError = new Error("Empty response from Gemini model");
        continue;
      }
    } catch (error) {
      console.error(`Gemini analysis error on attempt ${attempt}:`, error);
      lastError = error as Error;
      
      // Wait before retry (exponential backoff)
      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      }
    }
  }
  
  throw new Error(`Failed to analyze ocular height after ${maxRetries} attempts: ${lastError?.message}`);
}