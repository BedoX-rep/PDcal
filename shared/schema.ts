import { pgTable, text, serial, integer, real, timestamp, uuid, decimal, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const measurements = pgTable("measurements", {
  id: serial("id").primaryKey(),
  userId: uuid("user_id").notNull(),
  measurementName: text("measurement_name"),
  pdValue: decimal("pd_value", { precision: 5, scale: 2 }).notNull(),
  leftPupilX: integer("left_pupil_x").notNull(),
  leftPupilY: integer("left_pupil_y").notNull(),
  rightPupilX: integer("right_pupil_x").notNull(),
  rightPupilY: integer("right_pupil_y").notNull(),
  noseBridgeX: integer("nose_bridge_x"),
  noseBridgeY: integer("nose_bridge_y"),
  leftMonocularPd: decimal("left_monocular_pd", { precision: 5, scale: 2 }),
  rightMonocularPd: decimal("right_monocular_pd", { precision: 5, scale: 2 }),
  pixelDistance: decimal("pixel_distance", { precision: 8, scale: 2 }).notNull(),
  scaleFactor: decimal("scale_factor", { precision: 8, scale: 4 }).notNull(),
  originalImageUrl: text("original_image_url"),
  processedImageUrl: text("processed_image_url"),
  imageUrl: text("image_url"),
  isSaved: boolean("is_saved").default(false),
  leftOcularHeight: decimal("left_ocular_height", { precision: 5, scale: 2 }),
  rightOcularHeight: decimal("right_ocular_height", { precision: 5, scale: 2 }),
  ocularConfidence: decimal("ocular_confidence", { precision: 3, scale: 2 }),
  analysisNotes: text("analysis_notes"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMeasurementSchema = createInsertSchema(measurements).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const updateMeasurementSchema = createInsertSchema(measurements).omit({
  id: true,
  userId: true,
  createdAt: true,
}).partial();

export type InsertMeasurement = z.infer<typeof insertMeasurementSchema>;
export type UpdateMeasurement = z.infer<typeof updateMeasurementSchema>;
export type Measurement = typeof measurements.$inferSelect;
