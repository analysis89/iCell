#ifndef __RFsample_h
#define __RFsample_h

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
    class RFsample : public ImageToImageFilter< TImage, TImage >

    {
        public:
            /** Standard class typedefs. */
            typedef RFsample Self;
            typedef ImageToImageFilter< TImage, TImage > Superclass;
            typedef SmartPointer< Self > Pointer;

            typedef TImage InputImageType;
            typedef typename InputImageType::Pointer InputImagePointer;
            typedef typename InputImageType::RegionType InputImageRegionType;

            /** Method for creation through the object factory. */
            itkNewMacro(Self);

            /** Run-time type information (and related methods). */
            itkTypeMacro(RFsample, ImageToImageFilter);

            /** We need to override this method because of multiple input types */
            void GenerateInputRequestedRegion();

            /** The image to be sampled for training Random Forest Classifier.*/
            void SetInputImage(const InputImagePointer image);

            /** The segmentation for training Random Forest Classifer  **/
            void SetInputSeg(const InputImagePointer imgSeg);

            /** The number of components **/
            void SetNComp(const unsigned short nComp);

            /** The sameple data **/
            std::vector<std::vector<float> > GetSamples();

            /** The label data **/
            std::vector<float> GetLabels();

            /** The sample size **/
            unsigned long GetSize();


        protected:
            RFsample();
            ~RFsample(){}

            /** Does the real work. */
            virtual void GenerateData();

            /** Member attributes **/
            typedef std::vector<float> FloatVec;
            typedef std::vector< FloatVec > SampleArray;
            SampleArray m_Samples;
            std::vector<float> m_Labels;
            unsigned short m_nComp;

        private:
            RFsample(const Self &); //purposely not implemented
            void operator=(const Self &);  //purposely not implemented

            // Feature image
            std::vector<InputImagePointer> m_FeatureImages;

            // Segmentation image
            InputImagePointer m_LabelImage;

    };

} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "RFsample.txx"
#endif
#endif // __RFsample_h
