#ifndef __RFapply_h
#define __RFapply_h

#include "itkImageToImageFilter.h"

#include "classification.h"
#include "data.h"
#include "forest.h"

#include <iostream>
#include <ostream>

namespace itk
{
    template< class TImage>
    class RFapply : public ImageToImageFilter< TImage, TImage >

    {
        public:
          /** Standard class typedefs. */
          typedef RFapply Self;
          typedef ImageToImageFilter< TImage, TImage > Superclass;
          typedef SmartPointer< Self > Pointer;

          typedef TImage InputImageType;
          typedef typename InputImageType::Pointer InputImagePointer;
          typedef typename InputImageType::RegionType InputImageRegionType;
          typedef typename InputImageType::IndexType InputImageIndexType;
          typedef typename InputImageType::SizeType InputImageSizeType;

          /** Method for creation through the object factory. */
          itkNewMacro(Self);

          /** Run-time type information (and related methods). */
          itkTypeMacro(RFapply, ImageToImageFilter);

          /** We need to override this method because of multiple input types */
          void GenerateInputRequestedRegion();

          /** The image to be sampled for training Random Forest Classifier.*/
          void SetInputImage(const InputImagePointer image);
          void SetDummyImage(const InputImagePointer dummy);

          /** The forest binary filename.*/
          void SetForestFileName(const std::string forestFileName);

          /** The number of components **/
          void SetNComp(const unsigned short nComp);

          /** The number of components **/
          void SetNClass(const unsigned short nClass);

          /** Get the testing samples **/

        protected:
          RFapply();
          ~RFapply(){}

          /** Does the real work. */
          virtual void GenerateData();

          /** Member attributes **/
          typedef float GreyType;
          typedef float LabelType;
          typedef Histogram<GreyType, LabelType> HistogramType;
          typedef MLData<GreyType, HistogramType *> TestingDataType;

          typedef Histogram<GreyType, LabelType> RFHistogramType;
          typedef AxisAlignedClassifier<GreyType, LabelType> RFAxisClassifierType;
          typedef DecisionForest<RFHistogramType, RFAxisClassifierType, GreyType> RandomForestType;

          typedef Classification<GreyType, LabelType, RFAxisClassifierType> ClassificationType;
          ClassificationType classification;

          std::string m_forestFileName;
          unsigned short m_nComp;
          unsigned short m_nClass;

        private:
          RFapply(const Self &); //purposely not implemented
          void operator=(const Self &);  //purposely not implemented

          std::vector<InputImagePointer> m_FeatureImages;

    };

} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "RFapply.txx"
#endif


#endif // __RFapply_h
