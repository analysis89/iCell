#ifndef __RFsample_txx
#define __RFsample_txx
 
#include "RFsample.h"

#include "itkImageToImageFilter.h"
#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

#include "classification.h"
#include "data.h"
#include "forest.h"
 
namespace itk
{
    template< class TImage>
    RFsample<TImage>::RFsample()
    {
        m_nComp = 0;
    }

    template< class TImage>
    void RFsample<TImage>::SetInputImage(const InputImagePointer image)
    {
        m_FeatureImages.push_back(image);
        std::string name = "feature_";
        name += m_FeatureImages.size();
        this->ProcessObject::SetInput(name, image);
    }

    template< class TImage>
    void RFsample<TImage>::SetInputSeg(const InputImagePointer imgSeg)
    {
        m_LabelImage = imgSeg;
        this->SetInput(imgSeg);
    }

    template <class TImage>
    void RFsample<TImage>::SetNComp(const unsigned short nComp)
    {
        m_nComp = nComp;
    }

    template <class TImage>
    std::vector<std::vector<float> > RFsample<TImage>::GetSamples()
    {
        return m_Samples;
    }

    template <class TImage>
    std::vector<float> RFsample<TImage>::GetLabels()
    {
        return m_Labels;
    }

    template <class TImage>
    unsigned long RFsample<TImage>::GetSize()
    {
        return m_Labels.size();
    }

    template <class TImage>
    void RFsample<TImage>::GenerateInputRequestedRegion()
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
    void RFsample<TImage>::GenerateData()
    {
        /** Set the dummy output image **/
        typename TImage::Pointer dummyImage = this->GetOutput();
        dummyImage->SetRegions(m_FeatureImages[0]->GetRequestedRegion());
        dummyImage->CopyInformation(m_FeatureImages[0]);
        dummyImage->Allocate();

        // Set up an array of nComp iterators
        typedef itk::ImageRegionConstIterator<TImage> ConstIteratorType;
        std::vector<ConstIteratorType> CompIT;
        for(int i = 0; i < m_nComp; i++)
        {
            ConstIteratorType TrainIT(m_FeatureImages[i], m_FeatureImages[0]->GetRequestedRegion());
            CompIT.push_back(TrainIT);
        }
        // Loop over label IT
        ConstIteratorType labelIT(m_LabelImage, m_FeatureImages[0]->GetRequestedRegion());
        for(labelIT.GoToBegin(); !labelIT.IsAtEnd(); ++labelIT)
        {
            if (labelIT.Get() != 0)
            {
                FloatVec mysample(m_nComp, 0.0);
                for(int i = 0; i < m_nComp; i++)
                {
                    mysample[i] = CompIT[i].Get();
                }
                m_Samples.push_back(mysample);
                m_Labels.push_back(labelIT.Get());
            }
            for(int i = 0; i < m_nComp; i++)
            {
                ++CompIT[i];
            }
        }
    }
} // end namespace

#endif
