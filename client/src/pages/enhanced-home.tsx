import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileUpload } from "@/components/ui/file-upload";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { apiRequest } from "@/lib/queryClient";
import MeasurementResults from "./measurement-results";
import { MeasurementHistory } from "./measurement-history";
import { 
  Eye, 
  Info, 
  Shield, 
  CheckCircle, 
  AlertTriangle, 
  Ruler, 
  RotateCcw,
  Save,
  FileText,
  Edit,
  LogOut,
  User,
  Upload,
  History,
  Target,
  Zap
} from "lucide-react";
import type { Measurement } from "@shared/schema";

interface ProcessingResult {
  success: boolean;
  measurement: Measurement;
  result: {
    pd_value: number;
    left_pupil: { x: number; y: number };
    right_pupil: { x: number; y: number };
    pixel_distance: number;
    scale_factor: number;
    processed_image_path: string;
  };
}

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

export default function EnhancedHome() {
  const [activeTab, setActiveTab] = useState("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();
  const { user, signOut } = useAuth();

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('image', file);
      
      const response = await apiRequest('POST', '/api/measurements', formData);
      return response.json();
    },
    onSuccess: (data: ProcessingResult) => {
      setProcessingResult(data);
      setOcularHeightResult(null);
      setError(null);
      toast({
        title: "Success!",
        description: `Your PD has been measured: ${data.result.pd_value}mm`,
      });
    },
    onError: (error: any) => {
      console.error('Upload error:', error);
      setError(error.message || "Failed to process image");
      toast({
        title: "Processing Failed",
        description: error.message || "Could not detect pupils or AprilTag in your photo",
        variant: "destructive"
      });
    }
  });

  const ocularHeightMutation = useMutation({
    mutationFn: async (measurementId: number) => {
      try {
        const response = await apiRequest('POST', `/api/measurements/${measurementId}/ocular-height`, {});
        return response.json();
      } catch (error: any) {
        if (error.message && error.message.includes('Unexpected token')) {
          throw new Error('Measurement not found. Please upload a new image to get started.');
        }
        throw error;
      }
    },
    onSuccess: (data: OcularHeightResult) => {
      setOcularHeightResult(data);
      toast({
        title: "Ocular Height Analysis Complete!",
        description: `Left: ${data.ocularAnalysis.leftOcularHeight.toFixed(1)}mm, Right: ${data.ocularAnalysis.rightOcularHeight.toFixed(1)}mm`,
      });
    },
    onError: (error: any) => {
      console.error('Ocular height analysis error:', error);
      
      if (error.message && (error.message.includes('Measurement not found') || error.message.includes('404'))) {
        toast({
          title: "Measurement Expired",
          description: "Please upload a new image to use the AI ocular height analyzer.",
          variant: "destructive"
        });
        setProcessingResult(null);
        setOcularHeightResult(null);
        setSelectedFile(null);
      } else {
        toast({
          title: "Analysis Failed",
          description: error.message || "Could not analyze ocular height",
          variant: "destructive"
        });
      }
    }
  });

  const manualOcularHeightMutation = useMutation({
    mutationFn: async (data: { measurementId: number; leftFrameBottomY: number; rightFrameBottomY: number; zoomLevel: number; imageWidth: number; imageHeight: number }) => {
      try {
        const response = await apiRequest('POST', `/api/measurements/${data.measurementId}/manual-ocular-height`, {
          leftFrameBottomY: data.leftFrameBottomY,
          rightFrameBottomY: data.rightFrameBottomY,
          zoomLevel: data.zoomLevel,
          imageWidth: data.imageWidth,
          imageHeight: data.imageHeight
        });
        return response.json();
      } catch (error: any) {
        if (error.message && error.message.includes('Unexpected token')) {
          throw new Error('Measurement not found. Please upload a new image to get started.');
        }
        throw error;
      }
    },
    onSuccess: (data: OcularHeightResult) => {
      setOcularHeightResult(data);
      setShowImageEditor(false);
      toast({
        title: "Manual Ocular Height Calculated!",
        description: `Left: ${data.ocularAnalysis.leftOcularHeight.toFixed(1)}mm, Right: ${data.ocularAnalysis.rightOcularHeight.toFixed(1)}mm`,
      });
    },
    onError: (error: any) => {
      console.error('Manual ocular height error:', error);
      setShowImageEditor(false);
      
      if (error.message && (error.message.includes('Measurement not found') || error.message.includes('404'))) {
        toast({
          title: "Measurement Expired",
          description: "Please upload a new image to use the manual ocular height calculator.",
          variant: "destructive"
        });
        setProcessingResult(null);
        setOcularHeightResult(null);
        setSelectedFile(null);
      } else {
        toast({
          title: "Calculation Failed",
          description: error.message || "Could not calculate ocular height from manual line placement",
          variant: "destructive"
        });
      }
    }
  });

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setProcessingResult(null);
    setOcularHeightResult(null);
    setError(null);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setProcessingResult(null);
    setOcularHeightResult(null);
    setError(null);
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      uploadMutation.mutate(selectedFile);
    }
  };

  const handleNewMeasurement = () => {
    setSelectedFile(null);
    setProcessingResult(null);
    setOcularHeightResult(null);
    setError(null);
    setActiveTab("upload");
  };

  const handleAnalyzeOcularHeight = () => {
    if (processingResult?.measurement?.id) {
      ocularHeightMutation.mutate(processingResult.measurement.id);
    }
  };

  const handleManualOcularHeight = () => {
    setShowImageEditor(true);
  };

  const handleImageEditorSave = (leftLineY: number, rightLineY: number, zoomLevel: number, imageWidth: number, imageHeight: number) => {
    if (processingResult?.measurement?.id) {
      manualOcularHeightMutation.mutate({
        measurementId: processingResult.measurement.id,
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
  };

  const downloadAprilTag = () => {
    const link = document.createElement('a');
    link.href = '/src/assets/apriltag.svg';
    link.download = 'apriltag_50mm.svg';
    link.click();
    
    toast({
      title: "Download Started",
      description: "AprilTag SVG downloaded. Print at actual size (50mm x 50mm).",
    });
  };

  if (showImageEditor && processingResult) {
    return (
      <div className="min-h-screen bg-slate-50 p-4">
        <div className="max-w-7xl mx-auto">
          <div className="mb-4">
            <Button 
              variant="outline" 
              onClick={handleImageEditorCancel}
              className="flex items-center space-x-2"
            >
              <RotateCcw className="h-4 w-4" />
              <span>Back to Results</span>
            </Button>
          </div>
          <ImageEditor
            imageSrc={`/api/images/${processingResult.result.processed_image_path.split('/').pop() || ''}`}
            leftPupil={processingResult.result.left_pupil}
            rightPupil={processingResult.result.right_pupil}
            onSave={handleImageEditorSave}
            onCancel={handleImageEditorCancel}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Enhanced Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-slate-200/60 shadow-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                <Eye className="text-white text-xl" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  PD Measurement Pro
                </h1>
                <p className="text-sm text-slate-600">Precise Pupillary Distance Analysis</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-slate-600">
                <Shield className="h-4 w-4 text-green-600" />
                <span>Secure & Private</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-2 text-sm text-slate-700">
                  <User className="h-4 w-4" />
                  <span className="font-medium">{user?.email}</span>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={signOut}
                  className="flex items-center space-x-1 hover:bg-red-50 hover:text-red-700 hover:border-red-200"
                >
                  <LogOut className="h-4 w-4" />
                  <span>Sign Out</span>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="upload" className="flex items-center space-x-2">
              <Upload className="h-4 w-4" />
              <span>New Measurement</span>
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center space-x-2">
              <History className="h-4 w-4" />
              <span>My Measurements</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-8">
            {/* Enhanced Instructions */}
            <Card className="border-0 bg-gradient-to-r from-blue-50 to-purple-50 shadow-lg">
              <CardContent className="pt-6">
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center flex-shrink-0">
                    <Target className="h-6 w-6 text-blue-600" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-slate-800 mb-3">
                      How to Get Accurate PD Measurements
                    </h3>
                    <div className="grid md:grid-cols-3 gap-4 text-sm">
                      <div className="flex items-start space-x-2">
                        <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">1</div>
                        <div>
                          <p className="font-medium text-slate-700">Position yourself</p>
                          <p className="text-slate-600">Hold camera at arm's length, eyes level with lens</p>
                        </div>
                      </div>
                      <div className="flex items-start space-x-2">
                        <div className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs font-bold">2</div>
                        <div>
                          <p className="font-medium text-slate-700">Include AprilTag</p>
                          <p className="text-slate-600">Place the 50mm AprilTag near your face for scale</p>
                        </div>
                      </div>
                      <div className="flex items-start space-x-2">
                        <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs font-bold">3</div>
                        <div>
                          <p className="font-medium text-slate-700">Good lighting</p>
                          <p className="text-slate-600">Use natural light, avoid shadows on your face</p>
                        </div>
                      </div>
                    </div>
                    <div className="mt-4">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={downloadAprilTag}
                        className="flex items-center space-x-2"
                      >
                        <Download className="h-4 w-4" />
                        <span>Download AprilTag (50mm)</span>
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Enhanced Upload Section */}
            <Card className="shadow-lg border-0">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-xl">
                  <Zap className="h-5 w-5 text-yellow-500" />
                  <span>Upload Your Photo</span>
                </CardTitle>
                <CardDescription>
                  Select a clear frontal photo with good lighting and the AprilTag visible
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {!selectedFile ? (
                  <FileUpload 
                    selectedFile={selectedFile}
                    onFileSelect={handleFileSelect}
                    onRemoveFile={handleRemoveFile}
                  />
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                          <CheckCircle className="h-5 w-5 text-green-600" />
                        </div>
                        <div>
                          <p className="font-medium text-slate-800">{selectedFile.name}</p>
                          <p className="text-sm text-slate-500">
                            {(selectedFile.size / 1024 / 1024).toFixed(1)} MB
                          </p>
                        </div>
                      </div>
                      <Button variant="outline" size="sm" onClick={handleRemoveFile}>
                        Remove
                      </Button>
                    </div>

                    <div className="flex space-x-3">
                      <Button
                        onClick={handleAnalyze}
                        disabled={uploadMutation.isPending}
                        className="flex-1 h-12 text-base font-medium"
                      >
                        {uploadMutation.isPending ? (
                          <div className="flex items-center space-x-2">
                            <div className="animate-spin h-4 w-4 border-b-2 border-white rounded-full" />
                            <span>Processing...</span>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2">
                            <Ruler className="h-4 w-4" />
                            <span>Analyze PD</span>
                          </div>
                        )}
                      </Button>
                    </div>

                    {uploadMutation.isPending && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm text-slate-600">
                          <span>Processing image...</span>
                          <span>This may take a few seconds</span>
                        </div>
                        <Progress value={50} className="h-2" />
                      </div>
                    )}
                  </div>
                )}

                {error && (
                  <div className="flex items-start space-x-3 p-4 bg-red-50 rounded-lg border border-red-200">
                    <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-red-800">Processing Failed</p>
                      <p className="text-sm text-red-700 mt-1">{error}</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Enhanced Results Section */}
            {processingResult && (
              <Card className="shadow-lg border-0 bg-gradient-to-r from-green-50 to-blue-50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2 text-xl">
                    <CheckCircle className="h-6 w-6 text-green-600" />
                    <span>Measurement Results</span>
                  </CardTitle>
                  <CardDescription>
                    Your pupillary distance has been calculated successfully
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Main PD Result */}
                  <div className="text-center py-6 bg-white rounded-lg shadow-sm">
                    <div className="text-4xl font-bold text-slate-800 mb-2">
                      {processingResult.result.pd_value}mm
                    </div>
                    <p className="text-lg text-slate-600">Pupillary Distance</p>
                    <div className="flex justify-center space-x-4 mt-4 text-sm">
                      {processingResult.measurement.leftMonocularPd && (
                        <div>
                          <Badge variant="secondary" className="mb-1">Left Monocular</Badge>
                          <p className="font-medium">{processingResult.measurement.leftMonocularPd}mm</p>
                        </div>
                      )}
                      {processingResult.measurement.rightMonocularPd && (
                        <div>
                          <Badge variant="secondary" className="mb-1">Right Monocular</Badge>
                          <p className="font-medium">{processingResult.measurement.rightMonocularPd}mm</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Processed Image */}
                  <div className="bg-white rounded-lg shadow-sm p-4">
                    <h4 className="font-medium mb-3">Processed Image</h4>
                    <div className="aspect-video bg-slate-100 rounded-lg overflow-hidden">
                      <img 
                        src={`/api/images/${processingResult.result.processed_image_path.split('/').pop() || ''}`}
                        alt="Processed measurement result"
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </div>

                  {/* Ocular Height Analysis */}
                  <div className="bg-white rounded-lg shadow-sm p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="font-medium">Ocular Height Analysis</h4>
                      {ocularHeightResult ? (
                        <Badge className="bg-green-100 text-green-800">Complete</Badge>
                      ) : (
                        <Badge variant="outline">Optional</Badge>
                      )}
                    </div>

                    {ocularHeightResult ? (
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="text-center p-3 bg-blue-50 rounded-lg">
                          <div className="text-2xl font-bold text-blue-600">
                            {ocularHeightResult.ocularAnalysis.leftOcularHeight.toFixed(1)}mm
                          </div>
                          <p className="text-sm text-blue-700">Left Ocular Height</p>
                        </div>
                        <div className="text-center p-3 bg-purple-50 rounded-lg">
                          <div className="text-2xl font-bold text-purple-600">
                            {ocularHeightResult.ocularAnalysis.rightOcularHeight.toFixed(1)}mm
                          </div>
                          <p className="text-sm text-purple-700">Right Ocular Height</p>
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-slate-600 mb-4">
                        Measure the vertical distance from your pupils to your frame bottom line for precise lens fitting.
                      </p>
                    )}

                    <div className="flex space-x-3">
                      <Button
                        variant="outline"
                        onClick={handleAnalyzeOcularHeight}
                        disabled={ocularHeightMutation.isPending}
                        className="flex-1"
                      >
                        {ocularHeightMutation.isPending ? (
                          <div className="flex items-center space-x-2">
                            <div className="animate-spin h-3 w-3 border-b-2 border-current rounded-full" />
                            <span>Analyzing...</span>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2">
                            <Zap className="h-4 w-4" />
                            <span>AI Analysis</span>
                          </div>
                        )}
                      </Button>
                      <Button
                        variant="outline"
                        onClick={handleManualOcularHeight}
                        className="flex-1"
                      >
                        <Edit className="h-4 w-4 mr-2" />
                        Manual Measurement
                      </Button>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex space-x-3 pt-4">
                    <Button onClick={handleNewMeasurement} className="flex-1">
                      <Upload className="h-4 w-4 mr-2" />
                      New Measurement
                    </Button>
                    <Button 
                      variant="outline" 
                      onClick={() => setActiveTab("history")}
                      className="flex-1"
                    >
                      <History className="h-4 w-4 mr-2" />
                      View History
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="history">
            <MeasurementHistory onBackToUpload={() => setActiveTab("upload")} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}