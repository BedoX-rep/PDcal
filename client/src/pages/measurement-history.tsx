import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { ImageEditor } from "@/components/ui/image-editor";
import { 
  Eye, 
  Calendar, 
  Ruler, 
  ArrowLeft,
  FileText,
  Edit,
  Trash2,
  Download,
  CheckCircle,
  AlertTriangle,
  Clock
} from "lucide-react";
import { format } from "date-fns";
import type { Measurement } from "@shared/schema";

interface OcularHeightResult {
  success: boolean;
  measurement: Measurement;
  ocularAnalysis: {
    leftOcularHeight: number;
    rightOcularHeight: number;
    confidence: number;
    analysisNotes: string;
  };
}

interface MeasurementHistoryProps {
  onBackToUpload: () => void;
}

export function MeasurementHistory({ onBackToUpload }: MeasurementHistoryProps) {
  const [selectedMeasurement, setSelectedMeasurement] = useState<Measurement | null>(null);
  const [showImageEditor, setShowImageEditor] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: measurements, isLoading } = useQuery({
    queryKey: ['/api/measurements'],
    queryFn: async () => {
      const response = await apiRequest('GET', '/api/measurements');
      return response.json();
    },
  });

  const ocularHeightMutation = useMutation({
    mutationFn: async (measurementId: number) => {
      const response = await apiRequest('POST', `/api/measurements/${measurementId}/ocular-height`, {});
      return response.json();
    },
    onSuccess: (data: OcularHeightResult) => {
      queryClient.invalidateQueries({ queryKey: ['/api/measurements'] });
      toast({
        title: "Ocular Height Analysis Complete!",
        description: `Left: ${data.ocularAnalysis.leftOcularHeight.toFixed(1)}mm, Right: ${data.ocularAnalysis.rightOcularHeight.toFixed(1)}mm`,
      });
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Failed",
        description: error.message || "Could not analyze ocular height",
        variant: "destructive"
      });
    }
  });

  const manualOcularHeightMutation = useMutation({
    mutationFn: async (data: { measurementId: number; leftFrameBottomY: number; rightFrameBottomY: number; zoomLevel: number; imageWidth: number; imageHeight: number }) => {
      const response = await apiRequest('POST', `/api/measurements/${data.measurementId}/manual-ocular-height`, {
        leftFrameBottomY: data.leftFrameBottomY,
        rightFrameBottomY: data.rightFrameBottomY,
        zoomLevel: data.zoomLevel,
        imageWidth: data.imageWidth,
        imageHeight: data.imageHeight
      });
      return response.json();
    },
    onSuccess: (data: OcularHeightResult) => {
      queryClient.invalidateQueries({ queryKey: ['/api/measurements'] });
      setShowImageEditor(false);
      setSelectedMeasurement(null);
      toast({
        title: "Manual Ocular Height Calculated!",
        description: `Left: ${data.ocularAnalysis.leftOcularHeight.toFixed(1)}mm, Right: ${data.ocularAnalysis.rightOcularHeight.toFixed(1)}mm`,
      });
    },
    onError: (error: any) => {
      setShowImageEditor(false);
      toast({
        title: "Calculation Failed",
        description: error.message || "Could not calculate ocular height from manual line placement",
        variant: "destructive"
      });
    }
  });

  const deleteMutation = useMutation({
    mutationFn: async (measurementId: number) => {
      const response = await apiRequest('DELETE', `/api/measurements/${measurementId}`);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/measurements'] });
      toast({
        title: "Measurement Deleted",
        description: "Measurement has been removed from your history.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Delete Failed",
        description: error.message || "Could not delete measurement",
        variant: "destructive"
      });
    }
  });

  const handleAnalyzeOcularHeight = (measurementId: number) => {
    ocularHeightMutation.mutate(measurementId);
  };

  const handleManualOcularHeight = (measurement: Measurement) => {
    setSelectedMeasurement(measurement);
    setShowImageEditor(true);
  };

  const handleImageEditorSave = (leftLineY: number, rightLineY: number, zoomLevel: number, imageWidth: number, imageHeight: number) => {
    if (selectedMeasurement?.id) {
      manualOcularHeightMutation.mutate({
        measurementId: selectedMeasurement.id,
        leftFrameBottomY: leftLineY,
        rightFrameBottomY: rightLineY,
        zoomLevel,
        imageWidth,
        imageHeight
      });
    }
  };

  const handleImageEditorCancel = () => {
    setShowImageEditor(false);
    setSelectedMeasurement(null);
  };

  if (showImageEditor && selectedMeasurement) {
    return (
      <div className="min-h-screen bg-slate-50 p-4">
        <div className="max-w-7xl mx-auto">
          <div className="mb-4">
            <Button 
              variant="outline" 
              onClick={handleImageEditorCancel}
              className="flex items-center space-x-2"
            >
              <ArrowLeft className="h-4 w-4" />
              <span>Back to History</span>
            </Button>
          </div>
          <ImageEditor
            imageSrc={`/api/images/${selectedMeasurement.processedImageUrl?.split('/').pop() || ''}`}
            leftPupil={{ x: Number(selectedMeasurement.leftPupilX), y: Number(selectedMeasurement.leftPupilY) }}
            rightPupil={{ x: Number(selectedMeasurement.rightPupilX), y: Number(selectedMeasurement.rightPupilY) }}
            onSave={handleImageEditorSave}
            onCancel={handleImageEditorCancel}
          />
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading your measurements...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-800 mb-2">Measurement History</h1>
              <p className="text-muted-foreground">View and manage your pupillary distance measurements</p>
            </div>
            <Button onClick={onBackToUpload} className="flex items-center space-x-2">
              <Eye className="h-4 w-4" />
              <span>New Measurement</span>
            </Button>
          </div>
        </div>

        {!measurements || measurements.length === 0 ? (
          <Card className="text-center py-12">
            <CardContent>
              <div className="flex flex-col items-center space-y-4">
                <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center">
                  <FileText className="h-8 w-8 text-slate-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-800">No measurements yet</h3>
                  <p className="text-muted-foreground mt-1">Upload your first image to get started</p>
                </div>
                <Button onClick={onBackToUpload} className="mt-4">
                  <Eye className="h-4 w-4 mr-2" />
                  Take First Measurement
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {measurements.map((measurement: Measurement) => (
              <Card key={measurement.id} className="hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg flex items-center space-x-2">
                      <Ruler className="h-5 w-5 text-primary" />
                      <span>{measurement.pdValue}mm PD</span>
                    </CardTitle>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => deleteMutation.mutate(measurement.id)}
                      className="text-red-500 hover:text-red-700 hover:bg-red-50"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                  <CardDescription className="flex items-center space-x-2 text-sm">
                    <Calendar className="h-4 w-4" />
                    <span>{format(new Date(measurement.createdAt!), 'MMM d, yyyy \'at\' h:mm a')}</span>
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Processed Image */}
                  {measurement.processedImageUrl && (
                    <div className="aspect-video bg-slate-100 rounded-lg overflow-hidden">
                      <img 
                        src={`/api/images/${measurement.processedImageUrl.split('/').pop() || ''}`}
                        alt="Processed measurement"
                        className="w-full h-full object-contain"
                      />
                    </div>
                  )}

                  {/* Measurement Details */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    {measurement.leftMonocularPd && (
                      <div>
                        <p className="text-muted-foreground">Left Monocular</p>
                        <p className="font-medium">{measurement.leftMonocularPd}mm</p>
                      </div>
                    )}
                    {measurement.rightMonocularPd && (
                      <div>
                        <p className="text-muted-foreground">Right Monocular</p>
                        <p className="font-medium">{measurement.rightMonocularPd}mm</p>
                      </div>
                    )}
                  </div>

                  {/* Ocular Height Section */}
                  <div className="border-t pt-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-sm">Ocular Height</h4>
                      {measurement.leftOcularHeight && measurement.rightOcularHeight ? (
                        <Badge variant="secondary" className="bg-green-100 text-green-800">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Complete
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="bg-yellow-50 text-yellow-800 border-yellow-200">
                          <Clock className="h-3 w-3 mr-1" />
                          Pending
                        </Badge>
                      )}
                    </div>
                    
                    {measurement.leftOcularHeight && measurement.rightOcularHeight ? (
                      <div className="grid grid-cols-2 gap-4 text-sm mb-3">
                        <div>
                          <p className="text-muted-foreground">Left OH</p>
                          <p className="font-medium">{measurement.leftOcularHeight}mm</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Right OH</p>
                          <p className="font-medium">{measurement.rightOcularHeight}mm</p>
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground mb-3">
                        Ocular height not measured yet
                      </p>
                    )}

                    <div className="flex space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleAnalyzeOcularHeight(measurement.id)}
                        disabled={ocularHeightMutation.isPending}
                        className="flex-1"
                      >
                        {ocularHeightMutation.isPending ? (
                          <div className="animate-spin h-3 w-3 border-b-2 border-current rounded-full mr-1" />
                        ) : (
                          <Eye className="h-3 w-3 mr-1" />
                        )}
                        AI Analysis
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleManualOcularHeight(measurement)}
                        disabled={manualOcularHeightMutation.isPending}
                        className="flex-1"
                      >
                        <Edit className="h-3 w-3 mr-1" />
                        Manual
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}