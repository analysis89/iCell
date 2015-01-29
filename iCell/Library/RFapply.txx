#ifndef __RFapply_txx
#define __RFapply_txx
 
#include "RFapply.h"
 
#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

#include "classification.h"
#include "data.h"
#include "forest.h"

#include <iostream>
#include <ostream>
 
namespace itk
{
    template< class TImage>
    RFapply<TImage>::RFapply()
    {
        m_nComp = 0;
        m_nClass = 0;
    }

    template< class TImage>
    void RFapply<TImage>::SetInputImage(const InputImagePointer image)
    {
        m_FeatureImages.push_back(image);
        std::string name = "feature_";
        name += m_FeatureImages.size();
        this->ProcessObject::SetInput(name, image);
//        if (m_FeatureImages.size()==1)
//        {
//            this->SetInput(image);
//        }
    }

    template< class TImage>
    void RFapply<TImage>::SetDummyImage(const InputImagePointer dummy)
    {
        this->SetInput(dummy);
    }

    template <class TImage>
    void RFapply<TImage>::SetNComp(const unsigned short nComp)
    {
        m_nComp = nComp;
    }

    template <class TImage>
    void RFapply<TImage>::SetNClass(const unsigned short nClass)
    {
        m_nClass = nClass;
    }

    template <class TImage>
    void RFapply<TImage>::SetForestFileName(const std::string forestFileName)
    {
        m_forestFileName = forestFileName;
    }

    template <class TImage>
    void RFapply<TImage>::GenerateInputRequestedRegion()
    {
        itk::ImageSource<TImage>::GenerateInputRequestedRegion();

        for( itk::InputDataObjectIterator it(this); !it.IsAtEnd(); it++ )
        {
            // Check whether the input is an image of the appropriate dimension
            InputImageType *input = dynamic_cast<InputImageType*>(it.GetInput());
            InputImageRegionType inputRegion;
            this->CallCopyOutputRegionToInputRegion(inputRegion, this->GetOutput()->GetRequestedRegion());
            input->SetRequestedRegion(inputRegion);
        }
    }

    template< class TImage>
    void RFapply<TImage>::GenerateData()
    {
        // Set the testing sample data
        unsigned long size_xy = (m_FeatureImages[0]->GetRequestedRegion().GetSize()[0])*
                                (m_FeatureImages[0]->GetRequestedRegion().GetSize()[1]);
        TestingDataType testData(size_xy, m_nComp);

        // Fill in the testing sample data
        typedef itk::ImageRegionConstIterator<TImage> ConstIteratorType;
        for (int iComp = 0; iComp < m_nComp; iComp++)
        {
            unsigned long iTest = 0;
            ConstIteratorType testIT(m_FeatureImages[iComp], m_FeatureImages[0]->GetRequestedRegion());
            for (testIT.GoToBegin(); !testIT.IsAtEnd(); ++testIT)
            {
                testData.data[iTest][iComp] = testIT.Get();
                iTest++;
            }
        }

        // Setup the forest
        RandomForestType forest = RandomForestType(true);

        // Read the forest from file
        std::filebuf fb;
        fb.open(m_forestFileName, std::ios::binary | std::ios::in);
        if (fb.is_open())
        {
            std::istream fin(&fb);
            forest.Read(fin);
            fb.close();
        }
        else
        {
            std::cerr << "Error opening the file!" << std::endl;
        }

        // Setup soft predictions
        typedef ClassificationType::SoftPredictionT SoftPredictionType;
        SoftPredictionType softPrediction(size_xy, m_nClass);

        // Setup hard predictions
        typedef ClassificationType::HardPredictionT HardPredictionType;
        HardPredictionType hardPrediction;
        hardPrediction.Resize(size_xy);

        // Generate index-to-label mapping based on nClass
        std::map<std::size_t, LabelType> indexToLabelMap;
        indexToLabelMap.clear();
        for (int i = 0; i < m_nClass; i++)
        {
            indexToLabelMap.insert(std::map<index_t, int>::value_type(i, i));
        }
        bool are_labels_valid = true;

        // Get hard predictions
        classification.Predicting(forest, testData, are_labels_valid,
                                  indexToLabelMap, softPrediction, hardPrediction);

        // Reshape hard predictions into image
        typename TImage::Pointer testResult = this->GetOutput();
        testResult->SetRegions(m_FeatureImages[0]->GetRequestedRegion());
        testResult->CopyInformation(m_FeatureImages[0]);
        testResult->Allocate();

        typedef itk::ImageRegionIterator<TImage> IteratorType;
        IteratorType resultIT(testResult, testResult->GetRequestedRegion());
        unsigned long k = 0;
        for (resultIT.GoToBegin(); !resultIT.IsAtEnd(); ++resultIT)
        {
            testResult->SetPixel(resultIT.GetIndex(), hardPrediction[k]);
            k++;
        }
    }
} // end namespace
 
 
#endif
