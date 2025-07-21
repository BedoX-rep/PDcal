import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { FileUpload } from "@/components/ui/file-upload";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { ImageEditor } from "@/components/ui/image-editor";
import { 
  Eye, 
  Info, 
  Download, 
  Shield, 
  CheckCircle, 
  AlertTriangle, 
  Ruler, 
  RotateCcw,
  Save,
  FileText,
  Edit
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

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null);
  const [ocularHeightResult, setOcularHeightResult] = useState<OcularHeightResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showImageEditor, setShowImageEditor] = useState(false);
  const { toast } = useToast();

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('image', file);
      
      const response = await apiRequest('POST', '/api/measurements', formData);
      return response.json();
    },
    onSuccess: (data: ProcessingResult) => {
      setProcessingResult(data);
      setOcularHeightResult(null); // Reset ocular height when new image is processed
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
      const response = await apiRequest('POST', `/api/measurements/${measurementId}/ocular-height`, {});
      return response.json();
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
      toast({
        title: "Analysis Failed",
        description: error.message || "Could not analyze ocular height",
        variant: "destructive"
      });
    }
  });

  const manualOcularHeightMutation = useMutation({
    mutationFn: async (data: { measurementId: number; frameBottomY: number; zoomLevel: number; imageWidth: number; imageHeight: number }) => {
      const response = await apiRequest('POST', `/api/measurements/${data.measurementId}/manual-ocular-height`, {
        frameBottomY: data.frameBottomY,
        zoomLevel: data.zoomLevel,
        imageWidth: data.imageWidth,
        imageHeight: data.imageHeight
      });
      return response.json();
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
      toast({
        title: "Calculation Failed",
        description: error.message || "Could not calculate ocular height from manual line placement",
        variant: "destructive"
      });
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
  };

  const handleAnalyzeOcularHeight = () => {
    if (processingResult?.measurement?.id) {
      ocularHeightMutation.mutate(processingResult.measurement.id);
    }
  };

  const handleManualOcularHeight = () => {
    setShowImageEditor(true);
  };

  const handleImageEditorSave = (lineY: number, zoomLevel: number, imageWidth: number, imageHeight: number) => {
    if (processingResult?.measurement?.id) {
      manualOcularHeightMutation.mutate({
        measurementId: processingResult.measurement.id,
        frameBottomY: lineY,
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
    // Create a link to download the AprilTag SVG
    const link = document.createElement('a');
    link.href = '/src/assets/apriltag.svg';
    link.download = 'apriltag_50mm.svg';
    link.click();
    
    toast({
      title: "Download Started",
      description: "AprilTag SVG downloaded. Print at actual size (50mm x 50mm).",
    });
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
                <Eye className="text-white text-lg" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-slate-800">PD Measurement Tool</h1>
                <p className="text-sm text-slate-500">Accurate Pupillary Distance Calculator</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-slate-500">
              <Shield className="h-4 w-4 text-green-600" />
              <span>Secure & Private</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Instructions Card */}
        <Card className="mb-8 shadow-sm">
          <CardContent className="pt-6">
            <div className="flex items-start space-x-4">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                <Info className="h-5 w-5 text-primary" />
              </div>
              <div className="flex-1">
                <h2 className="text-lg font-semibold text-slate-800 mb-2">How to Take Your Photo</h2>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <div className="flex items-start space-x-2">
                      <div className="w-6 h-6 bg-primary text-white rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">1</div>
                      <p className="text-sm text-slate-600">Hold an AprilTag (50mm width) at forehead level, centered below your face</p>
                    </div>
                    <div className="flex items-start space-x-2">
                      <div className="w-6 h-6 bg-primary text-white rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">2</div>
                      <p className="text-sm text-slate-600">Look directly at the camera with eyes wide open</p>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-start space-x-2">
                      <div className="w-6 h-6 bg-primary text-white rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">3</div>
                      <p className="text-sm text-slate-600">Ensure good lighting and both pupils are clearly visible</p>
                    </div>
                    <div className="flex items-start space-x-2">
                      <div className="w-6 h-6 bg-primary text-white rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">4</div>
                      <p className="text-sm text-slate-600">Keep the camera at eye level and avoid tilting</p>
                    </div>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Download className="h-4 w-4 text-amber-600" />
                    <span className="text-sm font-medium text-amber-800">Need an AprilTag?</span>
                    <Button 
                      variant="link" 
                      className="text-sm text-primary hover:text-blue-700 underline p-0 h-auto"
                      onClick={downloadAprilTag}
                    >
                      Download & Print AprilTag
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Upload Section */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-slate-800 mb-4">Upload Your Photo</h2>
          <FileUpload 
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onRemoveFile={handleRemoveFile}
            isUploading={uploadMutation.isPending}
            uploadProgress={uploadMutation.isPending ? 75 : 0}
          />
          
          {selectedFile && !uploadMutation.isPending && !processingResult && !error && (
            <div className="mt-4">
              <Button 
                onClick={handleAnalyze}
                className="w-full bg-primary text-white hover:bg-blue-700 py-3"
                size="lg"
              >
                <Eye className="mr-2 h-4 w-4" />
                Analyze Photo PD and Ocular Height
              </Button>
            </div>
          )}
        </div>

        {/* Processing State */}
        {uploadMutation.isPending && (
          <Card className="mb-8 shadow-sm">
            <CardContent className="pt-6">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                  <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-800">Processing Your Photo</h3>
                  <p className="text-slate-500 mt-1">Detecting pupils and AprilTag...</p>
                </div>
                <Progress value={75} className="w-full max-w-md mx-auto" />
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error State */}
        {error && (
          <Card className="mb-8 border-red-200 shadow-sm">
            <CardContent className="pt-6">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto">
                  <AlertTriangle className="h-8 w-8 text-red-500" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-800">Detection Failed</h3>
                  <p className="text-slate-600 mt-1">{error}</p>
                </div>
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-left">
                  <h4 className="font-medium text-red-800 mb-2">Please check:</h4>
                  <ul className="text-sm text-red-700 space-y-1">
                    <li>• Both eyes are open and clearly visible</li>
                    <li>• AprilTag is positioned below your face and in focus</li>
                    <li>• Photo has good lighting and contrast</li>
                    <li>• No shadows covering your eyes or the AprilTag</li>
                  </ul>
                </div>
                <Button 
                  onClick={handleNewMeasurement}
                  className="bg-primary text-white hover:bg-blue-700"
                >
                  Try Another Photo
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Results Section */}
        {processingResult && (
          <Card className="shadow-sm">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-slate-800">Measurement Results</h2>
                <div className="flex items-center space-x-2 text-sm text-green-600">
                  <CheckCircle className="h-4 w-4" />
                  <span>Successfully Processed</span>
                </div>
              </div>

              <div className="grid lg:grid-cols-2 gap-8">
                {/* Processed Image Display */}
                <div className="space-y-4">
                  <h3 className="font-medium text-slate-700">Processed Image</h3>
                  <div className="relative">
                    <img 
                      src={`/api/images/${processingResult.result.processed_image_path}`}
                      alt="Processed photo with pupil and AprilTag detection overlays"
                      className="w-full h-auto rounded-lg border border-slate-200"
                    />
                    <div className="absolute top-4 left-4 bg-green-600 text-white px-2 py-1 rounded text-xs font-medium">
                      ✓ Pupils Detected
                    </div>
                    <div className="absolute top-4 right-4 bg-green-600 text-white px-2 py-1 rounded text-xs font-medium">
                      ✓ AprilTag Found
                    </div>
                  </div>
                  
                  <div className="flex justify-center space-x-4">
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="text-primary hover:text-blue-700"
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = `/api/images/${processingResult.result.processed_image_path}`;
                        link.download = `pd_result_${processingResult.result.processed_image_path}`;
                        link.click();
                      }}
                    >
                      <Download className="mr-1 h-3 w-3" />
                      Download Result
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={handleNewMeasurement}
                    >
                      <RotateCcw className="mr-1 h-3 w-3" />
                      New Measurement
                    </Button>
                  </div>
                </div>

                {/* Measurement Results */}
                <div className="space-y-6">
                  {/* PD Result Card */}
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6">
                    <div className="text-center">
                      <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto mb-4">
                        <Ruler className="text-white text-xl" />
                      </div>
                      <h3 className="text-lg font-semibold text-slate-800 mb-2">Your Pupillary Distance</h3>
                      <div className="text-4xl font-bold text-primary mb-2">
                        {processingResult.result.pd_value}
                      </div>
                      <p className="text-lg text-slate-600 font-medium">millimeters</p>
                      <div className="mt-4 text-sm text-slate-500">
                        <div className="flex items-center justify-center space-x-1">
                          <Info className="h-3 w-3" />
                          <span>Accuracy: ±0.5mm</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Measurement Details */}
                  <div className="space-y-4">
                    <h4 className="font-medium text-slate-700">Measurement Details</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center py-2 border-b border-slate-100">
                        <span className="text-sm text-slate-600">Left Pupil Position</span>
                        <span className="text-sm font-medium text-slate-800">
                          X: {processingResult.result.left_pupil.x}, Y: {processingResult.result.left_pupil.y}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-slate-100">
                        <span className="text-sm text-slate-600">Right Pupil Position</span>
                        <span className="text-sm font-medium text-slate-800">
                          X: {processingResult.result.right_pupil.x}, Y: {processingResult.result.right_pupil.y}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-slate-100">
                        <span className="text-sm text-slate-600">Pixel Distance</span>
                        <span className="text-sm font-medium text-slate-800">
                          {Math.round(processingResult.result.pixel_distance)} pixels
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-slate-100">
                        <span className="text-sm text-slate-600">AprilTag Size (Reference)</span>
                        <span className="text-sm font-medium text-slate-800">50.0 mm</span>
                      </div>
                      <div className="flex justify-between items-center py-2">
                        <span className="text-sm text-slate-600">Scale Factor</span>
                        <span className="text-sm font-medium text-slate-800">
                          {processingResult.result.scale_factor} mm/pixel
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Ocular Height Analysis */}
                  {!ocularHeightResult && (
                    <div className="space-y-3">
                      <Button 
                        className="w-full bg-gradient-to-r from-purple-600 to-purple-700 text-white hover:from-purple-700 hover:to-purple-800"
                        onClick={handleAnalyzeOcularHeight}
                        disabled={ocularHeightMutation.isPending}
                      >
                        {ocularHeightMutation.isPending ? (
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                        ) : (
                          <Eye className="mr-2 h-4 w-4" />
                        )}
                        {ocularHeightMutation.isPending ? "Analyzing Ocular Height..." : "Analyze Ocular Height with AI"}
                      </Button>
                      <Button 
                        variant="outline"
                        className="w-full border-purple-200 text-purple-700 hover:bg-purple-50"
                        onClick={handleManualOcularHeight}
                        disabled={manualOcularHeightMutation.isPending}
                      >
                        {manualOcularHeightMutation.isPending ? (
                          <div className="w-4 h-4 border-2 border-purple-600 border-t-transparent rounded-full animate-spin mr-2"></div>
                        ) : (
                          <Edit className="mr-2 h-4 w-4" />
                        )}
                        {manualOcularHeightMutation.isPending ? "Calculating..." : "Manual Ocular Height Calculation"}
                      </Button>
                      <p className="text-xs text-slate-500 text-center">
                        AI uses Gemini for automatic detection, Manual allows precise line placement with zoom
                      </p>
                    </div>
                  )}

                  {/* Ocular Height Results */}
                  {ocularHeightResult && (
                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-xl p-6 space-y-4">
                      <div className="text-center">
                        <div className="w-16 h-16 bg-gradient-to-r from-purple-600 to-purple-700 rounded-full flex items-center justify-center mx-auto mb-4">
                          <Eye className="text-white text-xl" />
                        </div>
                        <h3 className="text-lg font-semibold text-slate-800 mb-4">Ocular Height Analysis</h3>
                        
                        <div className="grid grid-cols-2 gap-4 mb-4">
                          <div className="text-center">
                            <div className="text-2xl font-bold text-purple-600 mb-1">
                              {ocularHeightResult.ocularAnalysis.leftOcularHeight.toFixed(1)}
                            </div>
                            <p className="text-sm text-slate-600 font-medium">Left Eye (mm)</p>
                          </div>
                          <div className="text-center">
                            <div className="text-2xl font-bold text-purple-600 mb-1">
                              {ocularHeightResult.ocularAnalysis.rightOcularHeight.toFixed(1)}
                            </div>
                            <p className="text-sm text-slate-600 font-medium">Right Eye (mm)</p>
                          </div>
                        </div>
                        
                        <div className="text-xs text-slate-500 mb-3">
                          Confidence: {Math.round(ocularHeightResult.ocularAnalysis.confidence * 100)}%
                        </div>
                        
                        <div className="bg-white/50 rounded-lg p-3 text-left">
                          <p className="text-xs text-slate-600 font-medium mb-1">AI Analysis Notes:</p>
                          <p className="text-xs text-slate-500">{ocularHeightResult.ocularAnalysis.analysisNotes}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Save/Export Options */}
                  <div className="space-y-3">
                    <Button 
                      className="w-full bg-primary text-white hover:bg-blue-700"
                      onClick={() => toast({ title: "Saved", description: "Measurement saved successfully" })}
                    >
                      <Save className="mr-2 h-4 w-4" />
                      Save Measurement
                    </Button>
                    <Button 
                      variant="outline" 
                      className="w-full"
                      onClick={() => toast({ title: "Export", description: "PDF report will be generated" })}
                    >
                      <FileText className="mr-2 h-4 w-4" />
                      Export PDF Report
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Image Editor Modal */}
        {showImageEditor && processingResult && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg max-w-5xl w-full max-h-[90vh] overflow-auto">
              <ImageEditor
                imageSrc={`/api/images/${processingResult.result.processed_image_path}`}
                leftPupil={processingResult.result.left_pupil}
                rightPupil={processingResult.result.right_pupil}
                onSave={handleImageEditorSave}
                onCancel={handleImageEditorCancel}
              />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 mt-12">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-4 text-sm text-slate-500">
              <div className="flex items-center space-x-1">
                <Shield className="h-4 w-4 text-green-600" />
                <span>Your photos are processed locally and not stored</span>
              </div>
            </div>
            <div className="flex items-center space-x-6 text-sm text-slate-500">
              <a href="#" className="hover:text-slate-700 transition-colors">Privacy Policy</a>
              <a href="#" className="hover:text-slate-700 transition-colors">Terms of Use</a>
              <a href="#" className="hover:text-slate-700 transition-colors">Support</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
