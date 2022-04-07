using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace INTEXPractice.Models
{
    public class InsuranceData
    {
        //public int InsuranceId { get; set; }
        public float Year { get; set; }
        public float Month { get; set; }
        public float Day { get; set; }
        public float Milepoint { get; set; }
        public float Intersection_Related { get; set; }
        public float  Teenage_Driver_Involved { get; set; }
        public float  Night_Dark_Condition { get; set; }
        public float Single_Vehicle { get; set; }

        public float Crash_Id { get; set; }

        public Tensor<float> AsTensor()
        {
            float[] data = new float[]
            {
                Year, Month, Day, Milepoint, Intersection_Related, Teenage_Driver_Involved, Night_Dark_Condition, Single_Vehicle
            };
            int[] dimensions = new int[] { 1, 8 };
            return new DenseTensor<float>(data, dimensions);
        }
    }
}
