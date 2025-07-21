import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRoute } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { ImageEditor } from "@/components/ui/image-editor";
import { Ruler, Save, Eye, Edit, CheckCircle, Target, ImageIcon, Info } from "lucide-react";
import { format } from "date-fns";
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
  const [showImageEditor, setShowImageEditor] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Fetch measurement if accessing via route
  const { data: fetchedMeasurement, isLoading } = useQuery({
    queryKey: ["/api/measurements", measurementId],
    enabled: !!measurementId && !measurementData,
  });

  // Use either passed data or fetched data
  const measurement = (measurementData?.measurement || fetchedMeasurement) as Measurement;
  const result = measurementData?.result;

  const saveMeasurementMutation = useMutation({
    mutationFn: async (name: string) => {
      if (!measurement?.id) throw new Error("No measurement to save");
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
      if (!measurement?.id) throw new Error("No measurement to update");
      const response = await apiRequest("POST", `/api/measurements/${measurement.id}/manual-ocular-height`, data);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Success", 
        description: "Ocular height calculated successfully!",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/measurements", measurement.id.toString()] });
      setShowImageEditor(false);
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
      if (!measurement?.id) throw new Error("No measurement to analyze");
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
        description: "AI analysis failed",
        variant: "destructive",
      });
    },
  });

  const handleSaveMeasurement = () => {
    saveMeasurementMutation.mutate(measurementName);
  };

  const handleManualOcularHeight = () => {
    setShowImageEditor(true);
  };

  const handleAIAnalysis = () => {
    aiAnalysisMutation.mutate();
  };

  const handleImageEditorSave = (leftLineY: number, rightLineY: number, zoomLevel: number, imageWidth: number, imageHeight: number) => {
    manualOcularHeightMutation.mutate({
      leftFrameBottomY: leftLineY,
      rightFrameBottomY: rightLineY,
      zoomLevel,
      imageWidth,
      imageHeight
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!measurement) {
    return (
      <Card>
        <CardContent className="pt-6 text-center">
          <div className="space-y-4">
            <Target className="h-16 w-16 text-muted-foreground mx-auto" />
            <div>
              <h3 className="text-lg font-medium">No measurement found</h3>
              <p className="text-muted-foreground">
                The measurement data could not be loaded
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Show image editor if active
  if (showImageEditor) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">Manual Ocular Height Measurement</h2>
            <p className="text-muted-foreground">Position the frame bottom line for accurate measurement</p>
          </div>
          <Button variant="outline" onClick={() => setShowImageEditor(false)}>
            Cancel
          </Button>
        </div>
        
        <ImageEditor
          imageSrc={`/api/images/${measurement.processedImageUrl?.split('/').pop() || ''}`}
          leftPupil={{ x: Number(measurement.leftPupilX), y: Number(measurement.leftPupilY) }}
          rightPupil={{ x: Number(measurement.rightPupilX), y: Number(measurement.rightPupilY) }}
          onSave={handleImageEditorSave}
          onCancel={() => setShowImageEditor(false)}
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Save Measurement Section */}
      <Card className="border-green-200 bg-green-50/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Save className="h-5 w-5" />
            Save Measurement
          </CardTitle>
          <CardDescription>
            Give your measurement a name and save it to your history
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <Label htmlFor="measurement-name">Measurement Name (Optional)</Label>
              <Input
                id="measurement-name"
                value={measurementName}
                onChange={(e) => setMeasurementName(e.target.value)}
                placeholder={`${measurement.pdValue}mm PD - ${format(new Date(), 'MMM d, yyyy')}`}
              />
            </div>
            <div className="flex items-end">
              <Button
                onClick={handleSaveMeasurement}
                disabled={saveMeasurementMutation.isPending}
                className="flex items-center gap-2"
              >
                {saveMeasurementMutation.isPending ? (
                  <div className="animate-spin h-4 w-4 border-b-2 border-current rounded-full" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                Save
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Results */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* PD Measurement Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Ruler className="h-5 w-5" />
              Pupillary Distance
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              <div className="text-4xl font-bold text-primary mb-2">
                {measurement.pdValue}mm
              </div>
              <p className="text-muted-foreground">Total PD</p>
            </div>
            
            <Separator />
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Pixel Distance:</span>
                <span className="text-sm font-medium">{measurement.pixelDistance}px</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Scale Factor:</span>
                <span className="text-sm font-medium">{measurement.scaleFactor}</span>
              </div>
              
              {measurement.leftMonocularPd && measurement.rightMonocularPd && (
                <>
                  <Separator />
                  <div className="space-y-2">
                    <h4 className="font-medium">Monocular PD</h4>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Left:</span>
                      <span className="text-sm font-medium">{measurement.leftMonocularPd}mm</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Right:</span>
                      <span className="text-sm font-medium">{measurement.rightMonocularPd}mm</span>
                    </div>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Ocular Height Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Ocular Height
            </CardTitle>
            <CardDescription>
              Measure the distance from pupil center to frame bottom
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {measurement.leftOcularHeight && measurement.rightOcularHeight ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="bg-green-100 text-green-800">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    Complete
                  </Badge>
                  {measurement.analysisNotes && (
                    <Badge variant="outline">{measurement.analysisNotes}</Badge>
                  )}
                  {measurement.ocularConfidence && (
                    <Badge variant="outline">
                      {Math.round(measurement.ocularConfidence * 100)}% confidence
                    </Badge>
                  )}
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">
                      {measurement.leftOcularHeight}mm
                    </div>
                    <p className="text-sm text-muted-foreground">Left OH</p>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">
                      {measurement.rightOcularHeight}mm
                    </div>
                    <p className="text-sm text-muted-foreground">Right OH</p>
                  </div>
                </div>
                
                {measurement.analysisNotes && (
                  <div className="bg-muted/50 p-3 rounded-lg">
                    <p className="text-sm">{measurement.analysisNotes}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <p className="text-muted-foreground text-sm">
                  Ocular height measurement is required for complete fitting data.
                  Choose an analysis method below.
                </p>
                
                <div className="grid gap-3">
                  <Button
                    onClick={handleAIAnalysis}
                    disabled={aiAnalysisMutation.isPending}
                    className="flex items-center gap-2"
                  >
                    {aiAnalysisMutation.isPending ? (
                      <div className="animate-spin h-4 w-4 border-b-2 border-current rounded-full" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                    AI Analysis
                  </Button>
                  
                  <Button
                    variant="outline"
                    onClick={handleManualOcularHeight}
                    disabled={manualOcularHeightMutation.isPending}
                    className="flex items-center gap-2"
                  >
                    <Edit className="h-4 w-4" />
                    Manual Measurement
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Processed Image */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ImageIcon className="h-5 w-5" />
            Processed Image
          </CardTitle>
          <CardDescription>
            View the analyzed image with detected pupils and measurements
          </CardDescription>
        </CardHeader>
        <CardContent>
          {measurement.processedImageUrl ? (
            <div className="aspect-video bg-muted rounded-lg overflow-hidden">
              <img
                src={`/api/images/${measurement.processedImageUrl.split('/').pop() || ''}`}
                alt="Processed measurement"
                className="w-full h-full object-contain"
              />
            </div>
          ) : (
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
              <p className="text-muted-foreground">No processed image available</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Measurement Info */}
      <Card className="border-blue-200 bg-blue-50/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Info className="h-5 w-5" />
            Measurement Details
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Measurement ID:</span>
            <span className="font-medium">#{measurement.id}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Created:</span>
            <span className="font-medium">
              {measurement.createdAt && format(new Date(measurement.createdAt), 'MMM d, yyyy \'at\' h:mm a')}
            </span>
          </div>
          {measurement.updatedAt && measurement.updatedAt !== measurement.createdAt && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Last Updated:</span>
              <span className="font-medium">
                {format(new Date(measurement.updatedAt), 'MMM d, yyyy \'at\' h:mm a')}
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}