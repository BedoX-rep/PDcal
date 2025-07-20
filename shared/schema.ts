import { pgTable, text, serial, integer, real, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const measurements = pgTable("measurements", {
  id: serial("id").primaryKey(),
  originalFilename: text("original_filename").notNull(),
  processedImagePath: text("processed_image_path"),
  pdValue: real("pd_value"),
  leftPupilX: real("left_pupil_x"),
  leftPupilY: real("left_pupil_y"),
  rightPupilX: real("right_pupil_x"),
  rightPupilY: real("right_pupil_y"),
  pixelDistance: real("pixel_distance"),
  scaleFactor: real("scale_factor"),
  apriltagDetected: integer("apriltag_detected").$type<boolean>(),
  pupilsDetected: integer("pupils_detected").$type<boolean>(),
  leftOcularHeight: real("left_ocular_height"),
  rightOcularHeight: real("right_ocular_height"),
  ocularHeightAnalyzed: integer("ocular_height_analyzed").$type<boolean>(),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertMeasurementSchema = createInsertSchema(measurements).omit({
  id: true,
  createdAt: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type InsertMeasurement = z.infer<typeof insertMeasurementSchema>;
export type Measurement = typeof measurements.$inferSelect;
