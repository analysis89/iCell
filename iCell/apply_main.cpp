#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRGBPixel.h"
#include "itkLaplacianImageFilter.h"
#include "itkImageAdaptor.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkLaplacianImageFilter.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkBilateralImageFilter.h"
#include "itkStreamingImageFilter.h"
#include "QuickView.h"

#include "Library/classification.h"
#include "Library/data.h"
#include "Library/RFapply.h"
#include "Library/forest.h"

#include "ImageCollectionToImageFilter.h"
#include "itkImageRegionIterator.h"


using namespace std;

class RedChannelPixelAccessor
{
public:
    typedef itk::RGBPixel<float> InternalType;
    typedef float ExternalType;
    static ExternalType Get( const InternalType & input )
    {
        return static_cast<ExternalType>( input.GetRed() );
    }
};

class GreenChannelPixelAccessor
{
public:
    typedef itk::RGBPixel<float> InternalType;
    typedef float ExternalType;
    static ExternalType Get( const InternalType & input )
    {
        return static_cast<ExternalType>( input.GetGreen() );
    }
};

class BlueChannelPixelAccessor
{
public:
    typedef itk::RGBPixel<float> InternalType;
    typedef float ExternalType;
    static ExternalType Get( const InternalType & input )
    {
        return static_cast<ExternalType>( input.GetBlue() );
    }
};

int main(int argc, char *argv[])
{

    /*
     *      This method trains an RF classifier and saves
     *      the forest.dat file for use in classification
     *
     *      Requires two input arguments:
     *         -i   Input Testing Image
     *         -o   Output Filename
     *         -f   Input Forest Filename
     *         -nc  Number of Classes
     *         -sd  Number of Streaming Divisions
     *
                                                          */

    cerr << " \n\n\t\tiCell Apply \n\t\tby Hyo Min Lee \n\n" << endl;

    // Parse command line arguments
    string inputFilename = "";
    string outputFilename = "";
    string forestFilename = "";
    unsigned short nClass = 0;
    unsigned int nStream = 0;

    bool inputFilename_ = true;
    bool outputFilename_ = true;
    bool forestFilename_ = true;
    bool nClass_ = true;
    bool nStream_ = true;

    for (unsigned int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "-i") == 0)
        {
            if (inputFilename_)
            {
                inputFilename = argv[i+1];
                i++;
                inputFilename_ = false;
            }
            else
            {
                cerr << "ERROR: Cannot have multiple input images!" << endl;
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "-o") == 0)
        {
            if (outputFilename_)
            {
                outputFilename = argv[i+1];
                i++;
                outputFilename_ = false;
            }
            else
            {
                cerr << "ERROR: Cannot have multiple segmentation images!" << endl;
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "-f") == 0)
        {
            if (forestFilename_)
            {
                forestFilename = argv[i+1];
                i++;
                forestFilename_ = false;
            }
            else
            {
                cerr << "ERROR: Cannot have multiple forest files!" << endl;
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "-nc") == 0)
        {
            if (nClass_)
            {
                nClass = stoi(argv[i+1]);
                i++;
                nClass_ = false;
            }
            else
            {
                cerr << "ERROR: Cannot set # of classes multiple times!" << endl;
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "-sd") == 0)
        {
            if (nStream_)
            {
                nStream = stoi(argv[i+1]);
                i++;
                nStream_ = false;
            }
            else
            {
                cerr << "ERROR: Cannot set # of streaming divisions multiple times!" << endl;
                return EXIT_FAILURE;
            }
        }
    }

    // Verify command line arguments
    if (argc < 2)
    {
        cerr << "Usage: " << endl;
        cerr << argv[0] << " inputImageFile" << endl;
        return EXIT_FAILURE;
    }
    if (inputFilename_)
    {
        cerr << "ERROR: No input image specified!" << endl;
        return EXIT_FAILURE;
    }
    if (outputFilename_)
    {
        cerr << "ERROR: No segmentation image specified!" << endl;
        return EXIT_FAILURE;
    }
    if (forestFilename_)
    {
        cerr << "ERROR: No forest file specified!" << endl;
        return EXIT_FAILURE;
    }
    if (nClass_)
    {
        cerr << "ERROR: Number of classes should be specified!" << endl;
        return EXIT_FAILURE;
    }
    if (nStream_)
    {
        cerr << "Number of streaming division is not specified. \nProceeding with default value of 1." << endl;
        nStream = 1;
    }

    // Display the input parameters for verification
    cerr << "\nInput image: " << inputFilename << endl;
    cerr << "Output image: " << outputFilename << endl;
    cerr << "Forest filename: " << forestFilename << endl;
    cerr << "# of classes: " << nClass << endl;
    cerr << "# of stream divisions: " << nStream  << "\n" << endl;

    // Basic Parameter
    unsigned short nComp = 15;

    // Read input image
    typedef itk::RGBPixel<float> PixelType; // rgb
    typedef itk::Image<PixelType, 2> RGBImageType; // image type
    typedef itk::ImageFileReader<RGBImageType> readerType; // file reader type
    readerType::Pointer reader = readerType::New(); // reader object
    reader->SetFileName(inputFilename.c_str());

    // Separate the RGB image
    typedef itk::ImageAdaptor<RGBImageType, RedChannelPixelAccessor> RedAdaptorType;
    typedef itk::ImageAdaptor<RGBImageType, GreenChannelPixelAccessor> GreenAdaptorType;
    typedef itk::ImageAdaptor<RGBImageType, BlueChannelPixelAccessor> BlueAdaptorType;

    RedAdaptorType::Pointer redAdaptor = RedAdaptorType::New();
    GreenAdaptorType::Pointer greenAdaptor = GreenAdaptorType::New();
    BlueAdaptorType::Pointer blueAdaptor = BlueAdaptorType::New();

    redAdaptor->SetImage(reader->GetOutput());
    greenAdaptor->SetImage(reader->GetOutput());
    blueAdaptor->SetImage(reader->GetOutput());

    typedef itk::Image<float,2> ImageType;

    typedef itk::RescaleIntensityImageFilter<RedAdaptorType, ImageType> RedRescalerType;
    typedef itk::RescaleIntensityImageFilter<GreenAdaptorType, ImageType> GreenRescalerType;
    typedef itk::RescaleIntensityImageFilter<BlueAdaptorType, ImageType> BlueRescalerType;

    RedRescalerType::Pointer redRescaler = RedRescalerType::New();
    GreenRescalerType::Pointer greenRescaler = GreenRescalerType::New();
    BlueRescalerType::Pointer blueRescaler = BlueRescalerType::New();

    redRescaler->SetInput(redAdaptor);
    greenRescaler->SetInput(greenAdaptor);
    blueRescaler->SetInput(blueAdaptor);

    redRescaler->SetOutputMinimum(0);
    redRescaler->SetOutputMaximum(255);
    greenRescaler->SetOutputMinimum(0);
    greenRescaler->SetOutputMaximum(255);
    blueRescaler->SetOutputMinimum(0);
    blueRescaler->SetOutputMaximum(255);


    // ================   FEATURE GENERATION   ================
    // Gaussian Image Filter
    typedef itk::DiscreteGaussianImageFilter<ImageType,ImageType> gaussType;
    gaussType::Pointer gaussFilter1 = gaussType::New();
    gaussType::Pointer gaussFilter2 = gaussType::New();
    gaussType::Pointer gaussFilter3 = gaussType::New();
    gaussFilter1->SetInput(redRescaler->GetOutput());
    gaussFilter1->SetVariance(2.56);
    gaussFilter2->SetInput(greenRescaler->GetOutput());
    gaussFilter2->SetVariance(2.56);
    gaussFilter3->SetInput(blueRescaler->GetOutput());
    gaussFilter3->SetVariance(2.56);

    // BL Filter
    typedef itk::BilateralImageFilter<ImageType,ImageType> bilateralType;
    bilateralType::Pointer bilateralFilter1 = bilateralType::New();
    bilateralType::Pointer bilateralFilter2 = bilateralType::New();
    bilateralType::Pointer bilateralFilter3 = bilateralType::New();
    bilateralFilter1->SetInput(redRescaler->GetOutput());
    bilateralFilter2->SetInput(greenRescaler->GetOutput());
    bilateralFilter3->SetInput(blueRescaler->GetOutput());

    // Laplacian Filter
    typedef itk::LaplacianImageFilter<ImageType,ImageType> laplacianType;
    laplacianType::Pointer laplacianFilter1 = laplacianType::New();
    laplacianType::Pointer laplacianFilter2 = laplacianType::New();
    laplacianType::Pointer laplacianFilter3 = laplacianType::New();
    laplacianFilter1->SetInput(redRescaler->GetOutput());
    laplacianFilter2->SetInput(greenRescaler->GetOutput());
    laplacianFilter3->SetInput(blueRescaler->GetOutput());

    // Gradient Magnitude Filter
    typedef itk::GradientMagnitudeImageFilter<ImageType,ImageType> gradmagType;
    gradmagType::Pointer gradmagFilter1 = gradmagType::New();
    gradmagType::Pointer gradmagFilter2 = gradmagType::New();
    gradmagType::Pointer gradmagFilter3 = gradmagType::New();
    gradmagFilter1->SetInput(redRescaler->GetOutput());
    gradmagFilter2->SetInput(greenRescaler->GetOutput());
    gradmagFilter3->SetInput(blueRescaler->GetOutput());

    // Hessian Filter
    typedef itk::HessianRecursiveGaussianImageFilter<ImageType,ImageType> hessType;
    hessType::Pointer hessFilter1 = hessType::New();
    hessType::Pointer hessFilter2 = hessType::New();
    hessType::Pointer hessFilter3 = hessType::New();
    hessFilter1->SetInput(redRescaler->GetOutput());
    hessFilter2->SetInput(greenRescaler->GetOutput());
    hessFilter3->SetInput(blueRescaler->GetOutput());

    cerr << "Preprocessing Has Started..." << endl;

     // ================   APPLYING CLASSIFICATION   ================

    ImageType::Pointer Input[15] = {
                                    gaussFilter1->GetOutput(),
                                    gaussFilter2->GetOutput(),
                                    gaussFilter3->GetOutput(),
                                    bilateralFilter1->GetOutput(),
                                    bilateralFilter2->GetOutput(),
                                    bilateralFilter3->GetOutput(),
                                    laplacianFilter1->GetOutput(),
                                    laplacianFilter2->GetOutput(),
                                    laplacianFilter3->GetOutput(),
                                    gradmagFilter1->GetOutput(),
                                    gradmagFilter2->GetOutput(),
                                    gradmagFilter3->GetOutput(),
                                    hessFilter1->GetOutput(),
                                    hessFilter2->GetOutput(),
                                    hessFilter3->GetOutput()
                                   };

    // Declare and instantiate the RF sampling filter
    typedef itk::RFapply<ImageType> applyType;
    applyType::Pointer apply = applyType::New();
    apply->SetNComp(nComp);
    apply->SetNClass(nClass);
    apply->SetForestFileName(forestFilename);
    for (int i = 0; i < nComp; i++)
    {
        apply->SetInputImage(Input[i]);
    }
    apply->SetDummyImage(redRescaler->GetOutput());

    // Streaming
    typedef itk::StreamingImageFilter<ImageType, ImageType> StreamingFilterType;
    StreamingFilterType::Pointer streamingFilter = StreamingFilterType::New();
    streamingFilter->SetInput(apply->GetOutput());
    streamingFilter->SetNumberOfStreamDivisions(nStream);

    // Save the results to a .nii file
    typedef itk::ImageFileWriter<ImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(outputFilename);
    writer->SetInput(streamingFilter->GetOutput());
    writer->SetNumberOfStreamDivisions(nStream);
    writer->Update();

    cerr << "Saved the full segmentation as: " << outputFilename << endl;

    return EXIT_SUCCESS;
}
