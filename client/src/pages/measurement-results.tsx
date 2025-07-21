import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useRoute } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ImageEditor } from "@/components/ui/image-editor";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { Eye, Ruler, Target, Save, Edit2 } from "lucide-react";
import type { Measurement } from "@shared/schema";

interface MeasurementResultsProps {
  measurementData?: {
    success: boolean;
    measurement: Measurement;
    result: any;
  };
  onSave?: () => void;
}

export default function MeasurementResults({ measurementData, onSave }: MeasurementResultsProps) {
  const [match, params] = useRoute("/measurement/:id");
  const measurementId = params?.id;
  const [measurementName, setMeasurementName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Fetch measurement if accessing via route
  const { data: fetchedMeasurement, isLoading } = useQuery({
    queryKey: ["/api/measurements", measurementId],
    enabled: !!measurementId && !measurementData,
  });

  // Use either passed data or fetched data
  const measurement = measurementData?.measurement || fetchedMeasurement;
  const result = measurementData?.result;

  const saveMeasurementMutation = useMutation({
    mutationFn: async (name: string) => {
      const response = await apiRequest("PATCH", `/api/measurements/${measurement.id}`, {
        measurementName: name || undefined,
      });
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Success",
        description: "Measurement saved successfully!",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/measurements"] });
      if (onSave) onSave();
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: "Failed to save measurement",
        variant: "destructive",
      });
    },
  });

  const manualOcularHeightMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiRequest("POST", `/api/measurements/${measurement.id}/manual-ocular-height`, data);
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Success", 
        description: "Ocular height calculated successfully!",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/measurements", measurement.id.toString()] });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: "Failed to calculate ocular height",
        variant: "destructive",
      });
    },
  });

  const aiAnalysisMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", `/api/measurements/${measurement.id}/ocular-height`);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Success",
        description: "AI analysis completed successfully!",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/measurements", measurement.id.toString()] });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: "Failed to complete AI analysis",
        variant: "destructive",
      });
    },
  });

  const handleSaveMeasurement = async () => {
    if (!measurement) return;
    setIsSaving(true);
    try {
      await saveMeasurementMutation.mutateAsync(measurementName);
    } finally {
      setIsSaving(false);
    }
  };

  const handleManualOcularHeight = (data: any) => {
    manualOcularHeightMutation.mutate(data);
  };

  const handleAiAnalysis = () => {
    aiAnalysisMutation.mutate();
  };

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center">Loading measurement...</div>
      </div>
    );
  }

  if (!measurement) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center text-red-500">Measurement not found</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Measurement Results
        </h1>
        <p className="text-muted-foreground">
          {measurement.measurementName || "Unnamed Measurement"}
        </p>
      </div>

      {/* Save Measurement Section - Only show if not saved yet */}
      {!measurement.measurementName && measurementData && (
        <Card className="border-blue-200 bg-blue-50/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-blue-700">
              <Save className="h-5 w-5" />
              Save This Measurement
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="measurement-name">Measurement Name (Optional)</Label>
              <Input
                id="measurement-name"
                placeholder="e.g., My PD Measurement, Morning Check, etc."
                value={measurementName}
                onChange={(e) => setMeasurementName(e.target.value)}
              />
            </div>
            <Button 
              onClick={handleSaveMeasurement}
              disabled={isSaving}
              className="w-full"
            >
              {isSaving ? "Saving..." : "Save Measurement"}
            </Button>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Results Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Pupillary Distance Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              <div className="text-4xl font-bold text-primary mb-2">
                {measurement.pdValue}mm
              </div>
              <Badge variant="secondary" className="mb-4">
                Primary PD Measurement
              </Badge>
            </div>

            <Separator />

            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Left Pupil Position:</span>
                <span className="text-sm font-medium">
                  ({measurement.leftPupilX}, {measurement.leftPupilY})
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Right Pupil Position:</span>
                <span className="text-sm font-medium">
                  ({measurement.rightPupilX}, {measurement.rightPupilY})
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Pixel Distance:</span>
                <span className="text-sm font-medium">{measurement.pixelDistance}px</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Scale Factor:</span>
                <span className="text-sm font-medium">{measurement.scaleFactor}</span>
              </div>
            </div>

            {(measurement.leftMonocularPd || measurement.rightMonocularPd) && (
              <>
                <Separator />
                <div className="space-y-2">
                  <h4 className="font-medium">Monocular PD</h4>
                  {measurement.leftMonocularPd && (
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Left Eye:</span>
                      <span className="text-sm font-medium">{measurement.leftMonocularPd}mm</span>
                    </div>
                  )}
                  {measurement.rightMonocularPd && (
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Right Eye:</span>
                      <span className="text-sm font-medium">{measurement.rightMonocularPd}mm</span>
                    </div>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Ocular Height Results */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Ruler className="h-5 w-5" />
              Ocular Height Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {measurement.leftOcularHeight || measurement.rightOcularHeight ? (
              <div className="space-y-3">
                <div className="text-center">
                  <Badge variant="outline" className="mb-2">
                    {measurement.analysisNotes?.includes("Manual") ? "Manual Measurement" : "AI Analysis"}
                  </Badge>
                  {measurement.ocularConfidence && (
                    <div className="text-sm text-muted-foreground">
                      Confidence: {(parseFloat(measurement.ocularConfidence) * 100).toFixed(0)}%
                    </div>
                  )}
                </div>

                <Separator />

                {measurement.leftOcularHeight && (
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Left Eye Height:</span>
                    <span className="text-sm font-medium">{measurement.leftOcularHeight}mm</span>
                  </div>
                )}
                {measurement.rightOcularHeight && (
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Right Eye Height:</span>
                    <span className="text-sm font-medium">{measurement.rightOcularHeight}mm</span>
                  </div>
                )}

                {measurement.analysisNotes && (
                  <>
                    <Separator />
                    <div className="text-sm text-muted-foreground">
                      {measurement.analysisNotes}
                    </div>
                  </>
                )}
              </div>
            ) : (
              <div className="text-center space-y-4">
                <div className="text-muted-foreground mb-4">
                  No ocular height analysis performed yet
                </div>
                <div className="space-y-2">
                  <Button
                    onClick={handleAiAnalysis}
                    disabled={aiAnalysisMutation.isPending}
                    className="w-full"
                    variant="outline"
                  >
                    {aiAnalysisMutation.isPending ? "Analyzing..." : "AI Analysis"}
                  </Button>
                  <p className="text-xs text-muted-foreground">
                    Or use manual measurement below
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Processed Image with Manual Measurement */}
      {measurement.processedImageUrl && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Processed Image & Manual Measurement
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ImageEditor
              imageUrl={`/api/images/${measurement.processedImageUrl.split('/').pop()}`}
              leftPupil={{ x: measurement.leftPupilX, y: measurement.leftPupilY }}
              rightPupil={{ x: measurement.rightPupilX, y: measurement.rightPupilY }}
              onManualMeasurement={handleManualOcularHeight}
              disabled={manualOcularHeightMutation.isPending}
            />
          </CardContent>
        </Card>
      )}

      {/* Measurement Metadata */}
      <Card>
        <CardHeader>
          <CardTitle>Measurement Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Measurement ID:</span>
            <span>#{measurement.id}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Created:</span>
            <span>{new Date(measurement.createdAt).toLocaleString()}</span>
          </div>
          {measurement.updatedAt && measurement.updatedAt !== measurement.createdAt && (
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Last Updated:</span>
              <span>{new Date(measurement.updatedAt).toLocaleString()}</span>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}