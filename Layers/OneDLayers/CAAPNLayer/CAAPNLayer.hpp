#pragma once
#include <vector>
#include <Activations/ActivationFunction.hpp>

class CAAPNLayer
{
  ActivationFunction *AF;

  // generator weights for the weights of the network
  std::vector<std::vector<float>> gww;
  // generator bias for the weights of the network
  std::vector<float> gwb;

  // generator weights for the bias of the network
  std::vector<std::vector<float>> gbw;
  // generator bias for the bias of the network
  std::vector<float> gbb;

  // sizes
  int input_size;
  int output_size;
  int weights_size;
  int bias_size;
  int generator_input_size;
  int generator_weights_for_weights_size;
  int generator_bias_for_weights_size;
  int generator_output_for_weights_size;
  int generator_weights_for_bias_size;
  int generator_bias_for_bias_size;
  int generator_output_for_bias_size;

  // stored for backpropagation
  std::vector<float> gx;
  std::vector<float> x;
  std::vector<std::vector<float>> wz;
  std::vector<std::vector<float>> w;
  std::vector<float> bz;
  std::vector<float> b;
  std::vector<float> z;

  // static consts
  float clamp_min = -10;
  float clamp_max = 10;

public:
  void Init(ActivationFunction *AF, int input_size, int output_size, float min, float max, float clamp_min, float clamp_max)
  {
    this->clamp_min = clamp_min;
    this->clamp_max = clamp_max;
    this->AF = AF;
    resize(input_size, output_size, min, max);
  }

  void resize(int input_size, int output_size, float min, float max)
  {
    this->input_size = input_size;
    this->output_size = output_size;
    this->weights_size = input_size * output_size;
    this->bias_size = output_size;

    this->generator_input_size = input_size + weights_size + bias_size;

    this->generator_output_for_weights_size = weights_size;
    this->generator_weights_for_weights_size = generator_input_size * generator_output_for_weights_size;
    this->generator_bias_for_weights_size = generator_output_for_weights_size;

    this->generator_output_for_bias_size = bias_size;
    this->generator_weights_for_bias_size = generator_input_size * generator_output_for_bias_size;
    this->generator_bias_for_bias_size = generator_output_for_bias_size;

    gww.resize(generator_output_for_weights_size, std::vector<float>(generator_input_size, 0));
    gwb.resize(generator_output_for_weights_size, 0);

    gbw.resize(generator_output_for_bias_size, std::vector<float>(generator_input_size, 0));
    gbb.resize(generator_output_for_bias_size, 0);

    // randomize weights
    for (int i = 0; i < gww.size(); i++)
    {
      for (int j = 0; j < gww[i].size(); j++)
      {
        gww[i][j] = Common::RandRange(min, max);
      }
    }
    // randomize bias
    for (int i = 0; i < gwb.size(); i++)
    {
      gwb[i] = Common::RandRange(min, max);
    }
    // randomize weights
    for (int i = 0; i < gbw.size(); i++)
    {
      for (int j = 0; j < gbw[i].size(); j++)
      {
        gbw[i][j] = Common::RandRange(min, max);
      }
    }
    // randomize bias
    for (int i = 0; i < gbb.size(); i++)
    {
      gbb[i] = Common::RandRange(min, max);
    }
  }

  std::vector<float> Forward(std::vector<float> input)
  {
    x = input;
    gx = input;
    std::vector<float> w_flat;
    if (w.size() == 0)
    {
      gx.resize(generator_input_size, 0);
      w_flat.resize(weights_size, 0);
    }
    else
    {
      Common::Flatten(w,w_flat);
      for (int i = 0; i < w_flat.size(); i++)
      {
        gx.push_back(w_flat[i]);
      }
      for (int i = 0; i < b.size(); i++)
      {
        gx.push_back(b[i]);
      }
    }

    Common::CalcZs(gx, gww, gwb, w_flat);
    Common::Split(w_flat, wz, x.size());
    w = AF->Activate(wz);

    Common::CalcZs(gx, gbw, gbb, bz);
    b = AF->Activate(bz);

    Common::CalcZs(x, w, b, z);

    return AF->Activate(z);
  }

  std::vector<float> Backward(std::vector<float> fg, float lr)
  {
    std::vector<float> dy_dz = AF->Derivative(z);
    
    std::vector<float> dz_dx;
    Common::DCalcZs_dx(w, dz_dx);

    std::vector<std::vector<float>> dz_dw;
    Common::DCalcZs_dw(x, w.size(), dz_dw);

    std::vector<float> dz_db;
    Common::DCalcZs_db(w.size(), dz_db);

    std::vector<std::vector<float>> dw_dwz = AF->Derivative(wz);

    std::vector<float> db_dbz = AF->Derivative(bz);

    std::vector<float> dwz_dgx;
    Common::DCalcZs_dx(gww, dwz_dgx);

    std::vector<std::vector<float>> dwz_dgww;
    Common::DCalcZs_dw(gx, gww.size(), dwz_dgww);

    std::vector<float> dwz_dgwb;
    Common::DCalcZs_db(gww.size(), dwz_dgwb);

    std::vector<float> dbz_dgx;
    Common::DCalcZs_dx(gbw, dbz_dgx);

    std::vector<std::vector<float>> dbz_dgbw;
    Common::DCalcZs_dw(gx, gbw.size(), dbz_dgbw);

    std::vector<float> dbz_dgbb;
    Common::DCalcZs_db(gbw.size(), dbz_dgbb);

    std::vector<float> dx(input_size, 0);
    for (int i = 0; i < input_size; i++)
    {
      for (int j = 0; j < output_size; j++)
      {
        dx[i] += dy_dz[j] * dz_dx[j]*lr;
        dx[i] += dy_dz[j] * dz_dw[j][i] * dw_dwz[j][i] * dwz_dgx[i]*lr;
        dx[i] += dy_dz[j] * dz_db[j] * db_dbz[j] * dbz_dgx[i]*lr;
      }
    }

    std::vector<std::vector<float>> dgww(generator_output_for_weights_size, std::vector<float>(generator_input_size, 0));
    std::vector<float> dgwb(generator_output_for_weights_size, 0);
    std::vector<std::vector<float>> dgbw(generator_output_for_bias_size, std::vector<float>(generator_input_size, 0));
    std::vector<float> dgbb(generator_output_for_bias_size, 0);
    for (int output_index = 0; output_index < output_size; output_index++)
    {
      for (int gen_input_index = 0; gen_input_index < generator_input_size; gen_input_index++)
      {
        for (int input_index = 0; input_index < input_size; input_index++)
        {
          int weight_index = input_index * output_size + output_index;
          dgww[weight_index][gen_input_index] += dy_dz[output_index] * dz_dw[output_index][input_index] * dw_dwz[output_index][input_index] * dwz_dgww[weight_index][gen_input_index]*lr;
          dgwb[weight_index] += dy_dz[output_index] * dz_dw[output_index][input_index] * dw_dwz[output_index][input_index] * dwz_dgwb[weight_index]*lr;
        }
        dgbw[output_index][gen_input_index] += dy_dz[output_index] * dz_db[output_index] * db_dbz[output_index] * dbz_dgbw[output_index][gen_input_index]*lr;
        dgbb[output_index] += dy_dz[output_index] * dz_db[output_index] * db_dbz[output_index] * dbz_dgbb[output_index]*lr;
      }
    }

    dgww=Common::Clamp(dgww, clamp_min, clamp_max);
    dgwb=Common::Clamp(dgwb, clamp_min, clamp_max);
    dgwb=Common::Clamp(dgwb, clamp_min, clamp_max);
    dgwb=Common::Clamp(dgwb, clamp_min, clamp_max);

    if (AF->IsTrainable())
    {
      AF->Backward(z, fg, lr, clamp_min, clamp_max);
      std::vector<float> wz_flat;
      Common::Flatten(wz,wz_flat);
      AF->Backward(wz_flat, fg, lr, clamp_min, clamp_max);
      AF->Backward(bz, fg, lr, clamp_min, clamp_max);
    }
    Common::Sub(gww, dgww, gww);
    Common::Sub(gwb, dgwb, gwb);
    Common::Sub(gbw, dgbw, gbw);
    Common::Sub(gbb, dgbb, gbb);
    

    return dx;
  }

  void RandomizeWeights(float min, float max)
  {
    for (int i = 0; i < gww.size(); i++)
    {
      for (int j = 0; j < gww[i].size(); j++)
      {
        gww[i][j] = Common::RandRange(min, max);
      }
    }
    for (int i = 0; i < gbw.size(); i++)
    {
      for (int j = 0; j < gbw[i].size(); j++)
      {
        gbw[i][j] = Common::RandRange(min, max);
      }
    }
  }

  void RandomizeBias(float min, float max)
  {
    for (int i = 0; i < gwb.size(); i++)
    {
      gwb[i] = Common::RandRange(min, max);
    }
    for (int i = 0; i < gbb.size(); i++)
    {
      gbb[i] = Common::RandRange(min, max);
    }
  }
};