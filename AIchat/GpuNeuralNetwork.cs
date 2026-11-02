using System;
using System.IO;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Neurony
{
    using View = ArrayView1D<float, Stride1D.Dense>;
    using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

    /// <summary>
    /// Stejná síť jako dřív (ReLU skryté vrstvy, sigmoid výstup, SGD po jednom vzorku),
    /// ale celý výpočet běží na NVIDIA GPU přes ILGPU/CUDA.
    /// Váhy jsou uložené transponovaně: W[vstup * pocetVystupu + vystup].
    /// </summary>
    public sealed class GpuNeuralNetwork : IDisposable
    {
        private const float LearningRate = 0.005f;

        private readonly Context context;
        private readonly Accelerator accelerator;
        private readonly int[] sizes;
        private readonly int layerCount;

        private readonly Buf[] weights;
        private readonly Buf[] biases;
        private readonly Buf[] outputs;
        private readonly Buf[] grads;
        private readonly Buf inputBuffer;

        private readonly Action<Index1D, View, View, View, View, int, int, int> forward;
        private readonly Action<Index1D, View, View, int> outputGrad;
        private readonly Action<Index1D, View, View, View, View, int> hiddenGrad;
        private readonly Action<Index1D, View, View, View, int, float> updateW;
        private readonly Action<Index1D, View, View, float> updateB;

        public int OutputSize => sizes[^1];
        public int InputSize => sizes[0];

        public GpuNeuralNetwork(int[] layerSizes)
        {
            sizes = (int[])layerSizes.Clone();
            layerCount = sizes.Length - 1;

            context = Context.Create(b => b.Cuda());
            accelerator = context.CreateCudaAccelerator(0);
            Console.WriteLine($"GPU: {accelerator.Name}");

            forward = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, View, View, View, View, int, int, int>(ForwardKernel);
            outputGrad = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, View, View, int>(OutputGradKernel);
            hiddenGrad = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, View, View, View, View, int>(HiddenGradKernel);
            updateW = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, View, View, View, int, float>(UpdateWeightsKernel);
            updateB = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, View, View, float>(UpdateBiasKernel);

            weights = new Buf[layerCount];
            biases = new Buf[layerCount];
            outputs = new Buf[layerCount];
            grads = new Buf[layerCount];
            inputBuffer = accelerator.Allocate1D<float>(sizes[0]);

            var random = new Random();

            for (int l = 0; l < layerCount; l++)
            {
                int inN = sizes[l];
                int outN = sizes[l + 1];

                float limit = MathF.Sqrt(2f / inN);
                float[] w = new float[inN * outN];
                for (int i = 0; i < w.Length; i++)
                    w[i] = (float)(random.NextDouble() * 2 - 1) * limit;

                weights[l] = accelerator.Allocate1D(w);
                biases[l] = accelerator.Allocate1D<float>(outN);
                outputs[l] = accelerator.Allocate1D<float>(outN);
                grads[l] = accelerator.Allocate1D<float>(outN);

                biases[l].MemSetToZero();
                outputs[l].MemSetToZero();
                grads[l].MemSetToZero();
            }
        }

        // ---------------------------------------------------------------- kernely

        private static void ForwardKernel(
            Index1D j, View input, View w, View b, View output,
            int inN, int outN, int sigmoid)
        {
            float sum = b[j];

            for (int i = 0; i < inN; i++)
            {
                float x = input[i];
                if (x != 0f)                       // vstup je skoro celý nulový
                    sum += x * w[i * outN + j];
            }

            if (sigmoid == 1)
            {
                if (sum > 30f) sum = 30f;
                if (sum < -30f) sum = -30f;
                output[j] = 1f / (1f + (float)Math.Exp(-sum));
            }
            else
            {
                output[j] = sum > 0f ? sum : 0f;
            }
        }

        private static void OutputGradKernel(Index1D j, View output, View grad, int targetId)
        {
            grad[j] = output[j] - (j == targetId ? 1f : 0f);
        }

        private static void HiddenGradKernel(
            Index1D i, View output, View grad, View nextGrad, View nextW, int nextOut)
        {
            float sum = 0f;

            for (int j = 0; j < nextOut; j++)
                sum += nextGrad[j] * nextW[i * nextOut + j];

            grad[i] = output[i] > 0f ? sum : 0f;
        }

        private static void UpdateWeightsKernel(
            Index1D idx, View w, View input, View grad, int outN, float lr)
        {
            int i = idx / outN;
            int j = idx % outN;
            float x = input[i];

            if (x != 0f)
                w[idx] -= lr * grad[j] * x;
        }

        private static void UpdateBiasKernel(Index1D j, View b, View grad, float lr)
        {
            b[j] -= lr * grad[j];
        }

        // ---------------------------------------------------------------- logika

        private void Forward(float[] input)
        {
            if (input.Length != sizes[0])
                throw new ArgumentException("Input size does not match network.");

            inputBuffer.View.CopyFromCPU(input);

            for (int l = 0; l < layerCount; l++)
            {
                View layerInput = l == 0 ? inputBuffer.View : outputs[l - 1].View;
                int isOutput = l == layerCount - 1 ? 1 : 0;

                forward(sizes[l + 1], layerInput, weights[l].View, biases[l].View,
                    outputs[l].View, sizes[l], sizes[l + 1], isOutput);
            }
        }

        public float[] Predict(float[] input)
        {
            Forward(input);
            return outputs[layerCount - 1].GetAsArray1D();
        }

        /// <summary>Jeden trénovací krok; cíl je one-hot na targetId.</summary>
        public void Train(float[] input, int targetId)
        {
            if (targetId < 0 || targetId >= OutputSize)
                throw new ArgumentException("Target out of range.");

            Forward(input);

            int last = layerCount - 1;
            outputGrad(sizes[last + 1], outputs[last].View, grads[last].View, targetId);

            // gradienty se počítají ze starých vah, proto až potom update
            for (int l = last - 1; l >= 0; l--)
            {
                hiddenGrad(sizes[l + 1], outputs[l].View, grads[l].View,
                    grads[l + 1].View, weights[l + 1].View, sizes[l + 2]);
            }

            for (int l = 0; l < layerCount; l++)
            {
                View layerInput = l == 0 ? inputBuffer.View : outputs[l - 1].View;
                int inN = sizes[l];
                int outN = sizes[l + 1];

                updateW(inN * outN, weights[l].View, layerInput, grads[l].View, outN, LearningRate);
                updateB(outN, biases[l].View, grads[l].View, LearningRate);
            }
        }

        // ---------------------------------------------------------------- ukládání

        public void Save(string filePath)
        {
            accelerator.Synchronize();

            using var writer = new BinaryWriter(File.Create(filePath));
            writer.Write(sizes.Length);
            foreach (int s in sizes) writer.Write(s);

            for (int l = 0; l < layerCount; l++)
            {
                WriteArray(writer, weights[l].GetAsArray1D());
                WriteArray(writer, biases[l].GetAsArray1D());
            }
        }

        public void Load(string filePath)
        {
            using var reader = new BinaryReader(File.OpenRead(filePath));

            int count = reader.ReadInt32();
            if (count != sizes.Length)
                throw new InvalidDataException("Network architecture mismatch.");

            for (int i = 0; i < count; i++)
                if (reader.ReadInt32() != sizes[i])
                    throw new InvalidDataException("Network architecture mismatch.");

            for (int l = 0; l < layerCount; l++)
            {
                weights[l].View.CopyFromCPU(ReadArray(reader, sizes[l] * sizes[l + 1]));
                biases[l].View.CopyFromCPU(ReadArray(reader, sizes[l + 1]));
            }
        }

        private static void WriteArray(BinaryWriter w, float[] data)
        {
            foreach (float f in data) w.Write(f);
        }

        private static float[] ReadArray(BinaryReader r, int length)
        {
            float[] data = new float[length];
            for (int i = 0; i < length; i++) data[i] = r.ReadSingle();
            return data;
        }

        public void Dispose()
        {
            accelerator.Dispose();
            context.Dispose();
        }
    }
}
