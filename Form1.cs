using Accord.Math.Geometry;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using LanguageExt;

namespace EmguCV
{
    public partial class Form1 : Form
    {
        private VideoCapture capture;
        private Mat frame;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            // Initialize the webcam capture
            capture = new VideoCapture();
            capture.ImageGrabbed += ProcessFrame;

            // Start the capture
            capture.Start();
        }

        /* ====================================================
         *  This function detect skin color of a given frame
         *  It use YCrCb color space defined as follow
         *          Y   => Luma component
         *          Cr  => Red difference
         *          Cb  => Blue difference
         * ================================================== */
        private Mat DetectSkin(Mat inputFrame)
        {
            // Convert to YCrCb
            Mat ycrcb = new Mat();
            CvInvoke.CvtColor(inputFrame, ycrcb, ColorConversion.Bgr2YCrCb);

            // Define the lower and upper bounds of the skin color in YCrCb
            ScalarArray lowerBound = new ScalarArray(new MCvScalar(0, 138, 67));
            ScalarArray upperBound = new ScalarArray(new MCvScalar(255, 173, 133));

            // Create a binary mask of the skin color
            Mat skinMask = new Mat();
            CvInvoke.InRange(ycrcb, lowerBound, upperBound, skinMask);

            // Matrix for a rectangle shape used as Kernel for morphological transformation
            Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));

            // Apply morphological operations to reduce noise
            CvInvoke.MorphologyEx(skinMask, skinMask, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(1));
            CvInvoke.MorphologyEx(skinMask, skinMask, MorphOp.Close, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(1));

            return skinMask;
        }

        private void DetectHand() 
        {
            // Retrieve the captured frame
            frame = new Mat();
            capture.Retrieve(frame);

            // ...
            Image<Bgr, Byte> finalImg = frame.ToImage<Bgr, Byte>().Flip(FlipType.Horizontal);
            Image<Gray, Byte> processingImg = finalImg.Convert<Gray, Byte>();
            processingImg = processingImg.ThresholdBinary(new Gray(150), new Gray(255));

            // Apply skin color segmentation
            // Mat skinMask = DetectSkin(frame);


            // Find contours in the binary image
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            //CvInvoke.FindContours(skinMask, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);
            CvInvoke.FindContours(processingImg, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            // Find the largest contour (The hand has to be in foreground otherwise it will detect face instead)
            double maxArea = 0;
            int maxAreaIndex = -1;

            for (int i = 0; i < contours.Size; i++)
            {
                double area = CvInvoke.ContourArea(contours[i]);
                if (area > maxArea)
                {
                    maxArea = area;
                    maxAreaIndex = i;
                }
            }


            // Draw the contour on the original frame
            if (maxAreaIndex != -1)
            {
                CvInvoke.DrawContours(frame, contours, maxAreaIndex, new MCvScalar(0, 255, 0), 2);
            }


            //defects points finding
            VectorOfInt hull = new VectorOfInt();
            Mat defects = new Mat();
            if (contours.Size > 0)
            {
                VectorOfPoint largestContour = new VectorOfPoint(contours[maxAreaIndex].ToArray());
                CvInvoke.ConvexHull(largestContour, hull, false, true);
                CvInvoke.ConvexityDefects(largestContour, hull, defects);
                if (!defects.IsEmpty)
                {
                    Matrix<int> m = new Matrix<int>(defects.Rows, defects.Cols, defects.NumberOfChannels);
                    defects.CopyTo(m);
                    Matrix<int>[] channels = m.Split();
                    for (int i = 1; i < defects.Rows; ++i)
                    {
                        finalImg.Draw(new System.Drawing.Point[] { largestContour[channels[0][i, 0]], largestContour[channels[1][i, 0]] }, new Bgr(100, 255, 100), 2);
                        CvInvoke.Circle(finalImg, new System.Drawing.Point(largestContour[channels[0][i, 0]].X, largestContour[channels[0][i, 0]].Y), 7, new MCvScalar(255, 0, 0), -1);
                    }
                }
            }




            // Display the frame in a PictureBox component
            // CamImageBox.Image = frame.ToImage<Bgr, byte>().ToBitmap();
            CamImageBox.Image = finalImg.ToBitmap();

            // Release the frame and contours to avoid memory leaks
            frame.Dispose();
            contours.Dispose();
        }


        private void Gesture() 
        {
            // retrieve frame from webcam
            frame = new Mat();
            capture.Retrieve(frame, 0);

            // gray & treshold
            Image<Bgr, Byte> finalImg = frame.ToImage<Bgr, Byte>().Flip(FlipType.Horizontal);
            Image<Gray, Byte> processingImg = finalImg.Convert<Gray, Byte>();
            processingImg = processingImg.ThresholdBinary(new Gray(128), new Gray(255));

            // morphological processing
            Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));
            processingImg.MorphologyEx(MorphOp.Erode, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Default, new MCvScalar());

            // edge detection
            Mat edges = new Mat(frame.Size, frame.Depth, 1);
            CvInvoke.Canny(processingImg, edges, 162, 255, 5);

            // contours finding
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            int largest_contour_index = 0;
            double largest_area = 0;
            MCvScalar redColor = new MCvScalar(0, 0, 255);
            CvInvoke.FindContours(edges, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);
            for (int i = 0; i < contours.Size; i++)
            {
                double a = CvInvoke.ContourArea(contours[i], false);
                if (a > largest_area)
                {
                    largest_area = a;
                    largest_contour_index = i;
                }
            }
            CvInvoke.DrawContours(finalImg, contours, largest_contour_index, redColor, 3, LineType.EightConnected, hierarchy);

            // Show processed img to picturebox
            CamImageBox.Image = finalImg.ToBitmap();
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            DetectHand();
        }
    
        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            // Release resources when the form is closing
            if (capture != null)
            {
                capture.Stop();
                capture.Dispose();
            }
        }
        private void btnStart_Click(object sender, EventArgs e)
        {

        }
    }
}