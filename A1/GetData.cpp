// Grab.cpp
/*
	Note: Before getting started, Basler recommends reading the Programmer's Guide topic
	in the pylon C++ API documentation that gets installed with pylon.
	If you are upgrading to a higher major version of pylon, Basler also
	strongly recommends reading the Migration topic in the pylon C++ API documentation.

	This sample illustrates how to grab and process images using the CInstantCamera class.
	The images are grabbed and processed asynchronously, i.e.,
	while the application is processing a buffer, the acquisition of the next buffer is done
	in parallel.

	The CInstantCamera class uses a pool of buffers to retrieve image data
	from the camera device. Once a buffer is filled and ready,
	the buffer can be retrieved from the camera object for processing. The buffer
	and additional image data are collected in a grab result. The grab result is
	held by a smart pointer after retrieval. The buffer is automatically reused
	when explicitly released or when the smart pointer object is destroyed.
*/

// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>
#ifdef PYLON_WIN_BUILD
#    include <pylon/PylonGUI.h>
#endif
#include<Windows.h>
#include <opencv2/opencv.hpp>
#include <pylon/BaslerUniversalInstantCamera.h>
#include <pylon/_BaslerUniversalCameraParams.h>
// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using cout.
using namespace std;
using namespace cv;
using namespace GenApi;
using namespace Basler_UniversalCameraParams;
// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 100;

int main(int argc, char* argv[])
{
	// The exit code of the sample application.
	int exitCode = 0;

	// Before using any pylon methods, the pylon runtime must be initialized.
	PylonInitialize();
	Mat frame;
	try
	{
		// Create an instant camera object with the camera device found first.
		CBaslerUniversalInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice());

		// Print the model name of the camera.
		cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

		// The parameter MaxNumBuffer can be used to control the count of buffers
		// allocated for grabbing. The default value of this parameter is 10.
		camera.MaxNumBuffer = 5;

		// Start the grabbing of c_countOfImagesToGrab images.
		// The camera device is parameterized with a default configuration which
		// sets up free-running continuous acquisition.

		camera.StartGrabbing(c_countOfImagesToGrab);

		//turn off ExposureAtuo and GainAuto
		camera.ExposureAuto.SetValue(ExposureAuto_Off);
		camera.GainAuto.SetValue(GainAuto_Off);
		camera.BalanceWhiteAuto.SetValue(BalanceWhiteAuto_Off);

		// This smart pointer will receive the grab result data.
		CGrabResultPtr ptrGrabResult;

		/// new image that convert to cv::Mat
		CImageFormatConverter formatConverter;
		formatConverter.OutputPixelFormat = PixelType_BGR8packed;
		CPylonImage pylonImage;

		// Camera.StopGrabbing() is called automatically by the RetrieveResult() method
		// when c_countOfImagesToGrab images have been retrieved.
		char c;

		//varieties
		double gainvalue = 10;//10x
		int exposuretime = 11000;//5000+1000x
		//picture no
		int no = 0;

		while (c = waitKey(1) && camera.IsGrabbing())
		{
			//set value
			camera.Gain.SetValue(gainvalue);
			camera.ExposureTime.SetValue(exposuretime);
			//Sleep(100);

			//change value
			no++;
			exposuretime += 1000;
			//gainvalue += 1;
			if (exposuretime > 20000 || gainvalue > 20)
				break;
			for (int i = 0; i < 10; i++) {
				// Wait for an image and then retrieve it. A timeout of 5000 ms is used.
				//camera.AcquisitionFrameCount.SetValue(3);
				camera.RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);
				// Image grabbed successfully?

				if (ptrGrabResult->GrabSucceeded())
				{
					cout << "Exp: " << exposuretime << " ";
					// Access the image data.
					cout << "Gain: " << gainvalue << " ";
					//cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
					//cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;

					const uint8_t *pImageBuffer = (uint8_t *)ptrGrabResult->GetBuffer();
					const size_t pImageBuffersize = (size_t)ptrGrabResult->GetImageSize();
					//calculate the average value of photos
					int totalgv = 0;
					
						for (int j = 0; j < pImageBuffersize; j++) {
						totalgv += (uint32_t)pImageBuffer[j];
					}
					cout << "avgvalue: " << (double)totalgv / pImageBuffersize << " "<<endl;
					
					//#ifdef PYLON_WIN_BUILD_)
									// Display the grabbed image.
									//Pylon::DisplayImage(1, ptrGrabResult);
					//#endif
									///convert to cv::Mat

					formatConverter.Convert(pylonImage, ptrGrabResult);
					frame = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t *)pylonImage.GetBuffer());
					resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));
					//save picture
					imshow("OpenCV Display Window", frame);
					string framename = "D:/OpenCV/hw/x64/Release/fixgain/";
					imwrite(framename + to_string(exposuretime) + "(" + to_string(i) + ").jpg", frame);

					/// show
					
					imshow("OpenCV Display Window", frame);
				}
				else
				{
					cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
				}
			}
		}
	}
	catch (const GenericException &e)
	{
		// Error handling.
		cerr << "An exception occurred." << endl
			<< e.GetDescription() << endl;
		exitCode = 1;
	}

	// Comment the following two lines to disable waiting on exit.
	cerr << endl << "Press Enter to exit." << endl;
	while (cin.get() != '\n');

	// Releases all pylon resources.
	PylonTerminate();

	return exitCode;
}

