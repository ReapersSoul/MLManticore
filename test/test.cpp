#include <MLManticore.hpp>
#include <iostream>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GraphicsGorilla/GraphicsGorilla.hpp>

float map(float x, float in_min, float in_max, float out_min, float out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

class TModel
{
public:
	virtual void Init(int input_size, int hidden_layer_size, int output_size, ActivationFunction *af, float min_weight_init, float max_weight_init, float min_clamp, float max_clamp) = 0;
	virtual std::vector<float> Forward(std::vector<float> input) = 0;
	virtual std::vector<float> Backward(std::vector<float> fg, float lr) = 0;
	virtual void Randomize() = 0;
};

class PerceptronLayerModel : public TModel
{
	PerceptronLayer layer1;
	PerceptronLayer layer2;
	PerceptronLayer layer3;
	float min_weight_init;
	float max_weight_init;
	float min_clamp;
	float max_clamp;

public:
	void Init(int input_size, int hidden_layer_size, int output_size, ActivationFunction *af, float min_weight_init, float max_weight_init, float min_clamp, float max_clamp)
	{
		this->min_weight_init = min_weight_init;
		this->max_weight_init = max_weight_init;
		this->min_clamp = min_clamp;
		this->max_clamp = max_clamp;
		layer1.Init(input_size, hidden_layer_size, af, min_weight_init, max_weight_init, min_clamp, max_clamp);
		layer2.Init(hidden_layer_size, hidden_layer_size, af, min_weight_init, max_weight_init, min_clamp, max_clamp);
		layer3.Init(hidden_layer_size, output_size, af, min_weight_init, max_weight_init, min_clamp, max_clamp);
	}

	std::vector<float> Forward(std::vector<float> input)
	{
		std::vector<float> output = layer1.Forward(input);
		output = layer2.Forward(output);
		output = layer3.Forward(output);
		return output;
	}

	std::vector<float> Backward(std::vector<float> fg, float lr)
	{
		std::vector<float> output = layer3.Backward(fg, lr);
		output = layer2.Backward(output, lr);
		output = layer1.Backward(output, lr);
		return output;
	}

	void Randomize()
	{
		layer1.RandomizeWeights(min_weight_init, max_weight_init);
		layer1.RandomizeBias(min_weight_init, max_weight_init);
		layer2.RandomizeWeights(min_weight_init, max_weight_init);
		layer2.RandomizeBias(min_weight_init, max_weight_init);
		layer3.RandomizeWeights(min_weight_init, max_weight_init);
		layer3.RandomizeBias(min_weight_init, max_weight_init);
	}
};

class RecurrentLayerModel : public TModel
{
	RecurrentLayer layer1;
	RecurrentLayer layer2;
	RecurrentLayer layer3;
	float min_weight_init;
	float max_weight_init;
	float min_clamp;
	float max_clamp;

public:
	void Init(int input_size, int hidden_layer_size, int output_size, ActivationFunction *af, float min_weight_init, float max_weight_init, float min_clamp, float max_clamp)
	{
		this->min_weight_init = min_weight_init;
		this->max_weight_init = max_weight_init;
		this->min_clamp = min_clamp;
		this->max_clamp = max_clamp;
		layer1.Init(input_size, hidden_layer_size, af, min_weight_init, max_weight_init, min_clamp, max_clamp);
		layer2.Init(hidden_layer_size, hidden_layer_size, af, min_weight_init, max_weight_init, min_clamp, max_clamp);
		layer3.Init(hidden_layer_size, output_size, af, min_weight_init, max_weight_init, min_clamp, max_clamp);
	}

	std::vector<float> Forward(std::vector<float> input)
	{
		std::vector<float> output = layer1.Forward(input);
		output = layer2.Forward(output);
		output = layer3.Forward(output);
		return output;
	}

	std::vector<float> Backward(std::vector<float> fg, float lr)
	{
		std::vector<float> output = layer3.Backward(fg, lr);
		output = layer2.Backward(output, lr);
		output = layer1.Backward(output, lr);
		return output;
	}

	void Randomize()
	{
		layer1.RandomizeWeights(min_weight_init, max_weight_init);
		layer1.RandomizeBias(min_weight_init, max_weight_init);
		layer2.RandomizeWeights(min_weight_init, max_weight_init);
		layer2.RandomizeBias(min_weight_init, max_weight_init);
		layer3.RandomizeWeights(min_weight_init, max_weight_init);
		layer3.RandomizeBias(min_weight_init, max_weight_init);
	}
};

class CAAPNModel : public TModel
{
	CAAPN layer1;
	CAAPN layer2;
	CAAPN layer3;
	float min_weight_init;
	float max_weight_init;
	float min_clamp;
	float max_clamp;

public:
	void Init(int input_size, int hidden_layer_size, int output_size, ActivationFunction *af, float min_weight_init, float max_weight_init, float min_clamp, float max_clamp)
	{
		this->min_weight_init = min_weight_init;
		this->max_weight_init = max_weight_init;
		this->min_clamp = min_clamp;
		this->max_clamp = max_clamp;
		layer1.Init(af, input_size, hidden_layer_size, min_weight_init, max_weight_init, min_clamp, max_clamp);
		layer2.Init(af, hidden_layer_size, hidden_layer_size, min_weight_init, max_weight_init, min_clamp, max_clamp);
		layer3.Init(af, hidden_layer_size, output_size, min_weight_init, max_weight_init, min_clamp, max_clamp);
	}

	std::vector<float> Forward(std::vector<float> input)
	{
		std::vector<float> output = layer1.Forward(input);
		output = layer2.Forward(output);
		output = layer3.Forward(output);
		return output;
	}

	std::vector<float> Backward(std::vector<float> fg, float lr)
	{
		std::vector<float> output = layer3.Backward(fg, lr);
		output = layer2.Backward(output, lr);
		output = layer1.Backward(output, lr);
		return output;
	}

	void Randomize()
	{
		layer1.RandomizeWeights(min_weight_init, max_weight_init);
		layer1.RandomizeBias(min_weight_init, max_weight_init);
		layer2.RandomizeWeights(min_weight_init, max_weight_init);
		layer2.RandomizeBias(min_weight_init, max_weight_init);
		layer3.RandomizeWeights(min_weight_init, max_weight_init);
		layer3.RandomizeBias(min_weight_init, max_weight_init);
	}
};

class NewCAAPN{
	ActivationFunction *af;
	DeepNeuralNetwork WeightGenerator;
	DeepNeuralNetwork BiasGenerator;
	std::vector<std::vector<float>> prevWeights;
	std::vector<float> prevBias;
	PerceptronLayer layer;
	int input_size;
	int output_size;
	int weights_size;
	int bias_size;
	float clamp_min;
	float clamp_max;
public:

	void Init(ActivationFunction *AF, int input_size, int output_size,std::vector<int> generatorHiddens, float min, float max, float clamp_min, float clamp_max)
	{
		this->clamp_min = clamp_min;
		this->clamp_max = clamp_max;
		this->af = AF;
		resize(input_size, output_size,generatorHiddens, min, max);
	}

	void resize(int input_size, int output_size,std::vector<int> generatorHiddens, float min, float max)
	{
		this->input_size = input_size;
		this->output_size = output_size;
		weights_size = input_size * output_size;
		bias_size = output_size;
		prevWeights.resize(input_size, std::vector<float>(output_size, 0));
		prevBias.resize(output_size, 0);
		layer.Init(input_size, output_size, af, min, max, clamp_min, clamp_max);
		std::vector<int> WeightGeneratorSizes = {input_size + weights_size + bias_size};
		std::vector<int> BiasGeneratorSizes = {input_size + weights_size + bias_size};
		for (int i = 0; i < generatorHiddens.size(); i++)
		{
			WeightGeneratorSizes.push_back(generatorHiddens[i]);
			BiasGeneratorSizes.push_back(generatorHiddens[i]);
		}
		WeightGeneratorSizes.push_back(weights_size);
		BiasGeneratorSizes.push_back(bias_size);

		WeightGenerator.Init(af,WeightGeneratorSizes, min, max, clamp_min, clamp_max);
		BiasGenerator.Init(af,BiasGeneratorSizes, min, max, clamp_min, clamp_max);
	}

	std::vector<float> Forward(std::vector<float> input)
	{
		std::vector<float> output;

		std::vector<float> gx(weights_size+input_size+bias_size, 0);
		for (int i = 0; i < input_size; i++)
		{
			gx[i] = input[i];
		}
		std::vector<float> prevWeightsFlat;
		Common::Flatten(prevWeights, prevWeightsFlat);
		for (int i = 0; i < weights_size; i++)
		{
			gx[input_size + i] = prevWeightsFlat[i];
		}
		for (int i = 0; i < bias_size; i++)
		{
			gx[input_size + weights_size + i] = prevBias[i];
		}
		Common::Split(WeightGenerator.Forward(gx), prevWeights, weights_size);
		prevBias = BiasGenerator.Forward(gx);

		layer.SetWeights(prevWeights);
		layer.SetBias(prevBias);
		output = layer.Forward(input);
		return output;
	}

	std::vector<float> Backward(std::vector<float> fg, float lr)
	{
		std::vector<float> dy_dx = layer.Backward(fg, lr);
		std::vector<std::vector<float>> dy_dw = layer.GetDWeights();
		std::vector<float> dy_dw_flat;
		Common::Flatten(dy_dw, dy_dw_flat);
		std::vector<float> dy_w_dgx = WeightGenerator.Backward(dy_dw_flat, lr);
		std::vector<float> dy_b_dgx = BiasGenerator.Backward(layer.GetDBias(), lr);
		std::vector<float> dx(input_size, 0);
		for (int i = 0; i < input_size; i++)
		{
			for (int j = 0; j < output_size; j++)
			{
				dx[i] += dy_dx[j] * lr;
				dx[i] += dy_w_dgx[i] * lr;
				dx[i] += dy_b_dgx[i] * lr;
			}
		}
		return dx;
	}

	void RandomizeWeights(float min, float max)
	{
		WeightGenerator.RandomizeWeights(min, max);
	}

	void RandomizeBias(float min, float max)
	{
		BiasGenerator.RandomizeWeights(min, max);
	}
};


class NewCAAPNModel : public TModel
{
	NewCAAPN layer1;
	NewCAAPN layer2;
	NewCAAPN layer3;
	float min_weight_init;
	float max_weight_init;
	float min_clamp;
	float max_clamp;

public:

	void Init(int input_size, int hidden_layer_size, int output_size, ActivationFunction *af, float min_weight_init, float max_weight_init, float min_clamp, float max_clamp)
	{
		this->min_weight_init = min_weight_init;
		this->max_weight_init = max_weight_init;
		this->min_clamp = min_clamp;
		this->max_clamp = max_clamp;
		std::vector<int> generatorHiddens = {10,5,10};
		layer1.Init(af, input_size, hidden_layer_size,generatorHiddens, min_weight_init, max_weight_init, min_clamp, max_clamp);
		layer2.Init(af, hidden_layer_size, hidden_layer_size,generatorHiddens, min_weight_init, max_weight_init, min_clamp, max_clamp);
		layer3.Init(af, hidden_layer_size, output_size,generatorHiddens, min_weight_init, max_weight_init, min_clamp, max_clamp);
	}

	std::vector<float> Forward(std::vector<float> input)
	{
		std::vector<float> output = layer1.Forward(input);
		output = layer2.Forward(output);
		output = layer3.Forward(output);
		return output;
	}

	std::vector<float> Backward(std::vector<float> fg, float lr)
	{
		std::vector<float> output = layer3.Backward(fg, lr);
		output = layer2.Backward(output, lr);
		output = layer1.Backward(output, lr);
		return output;
	}

	void Randomize()
	{
		layer1.RandomizeWeights(min_weight_init, max_weight_init);
		layer1.RandomizeBias(min_weight_init, max_weight_init);
		layer2.RandomizeWeights(min_weight_init, max_weight_init);
		layer2.RandomizeBias(min_weight_init, max_weight_init);
		layer3.RandomizeWeights(min_weight_init, max_weight_init);
		layer3.RandomizeBias(min_weight_init, max_weight_init);
	}
};

struct model_defintion
{
	std::string name;
	std::function<TModel *()> model;
	float lr;
};

struct ActivationFunctionDefinition
{
	std::string name;
	std::function<ActivationFunction *()> af;
	float lr_modifier;
};

struct LossFunctionDefinition
{
	std::string name;
	std::function<LossFunction *()> lf;
};

std::vector<model_defintion> models = {
	// {"PerceptronLayer", []()
	//  { return new PerceptronLayerModel(); }, .1},
	// {"RecurrentLayer", []()
	//  { return new RecurrentLayerModel(); }, .01},
	// {"CAAPN", []()
	//  { return new CAAPNModel(); }, 0.00001},
	{"NewCAAPN", []()
	 { return new NewCAAPNModel(); }, 0.00001}};

std::vector<ActivationFunctionDefinition> activation_functions = {
	{"Sigmoid", []()
	 { return new Sigmoid(); }, 1},
	{"ReLU", []()
	 { return new ReLU(); }, 1},
	{"LeakyReLU", []()
	 { return new LeakyReLU(); }, 1},
	{"Tanh", []()
	 { return new Tanh(); }, 1},
	{"GeLU", []()
	 { return new GeLU(); }, 1},
	{"Bezier", []()
	 { return new Bezier(20, -3, 3); }, .01},
	{"LerpEndToEnd", []()
	 { return new LerpEndToEnd(20, -1, 1); }, .001}};

std::vector<LossFunctionDefinition> loss_functions = {
	{"MeanSquaredError", []()
	 { return new MeanSquaredError(); }},
	{"CrossEntropy", []()
	 { return new CrossEntropy(); }},
	{"MeanAbsoluteError", []()
	 { return new MeanAbsoluteError(); }},
	{"LogLoss", []()
	 { return new LogLoss(); }}};

static GraphicsGorrila::TwoD::Text::Font font;

class Trainer
{
	ActivationFunction *af;
	TModel *model;
	LossFunction *lf;
	int af_index = 0;
	int lf_index = 0;
	int model_index = 0;

	float min_weight_init = -.000000000001;
	float max_weight_init = .000000000001;
	float min_clamp = -1;
	float max_clamp = 1;
	float lr = 0.001;
	int window_size = 1000;
	int history_size = 100;
	int input_size = 2;
	int hidden_layer_size = 10;
	int output_size = 2;

	std::vector<float> last_predicted_stick_x;
	std::vector<float> last_predicted_stick_y;
	std::vector<float> last_stick_x;
	std::vector<float> last_stick_y;

	std::vector<float> rolling_window;

	void Train(std::vector<float> input, std::vector<float> target, float lr)
	{
		std::vector<float> output = model->Forward(input);
		std::vector<float> fg = lf->Derivative(output, target);
		model->Backward(fg, lr);
	}

public:
	// ~Trainer()
	// {
	// 	delete model;
	// 	delete af;
	// 	delete lf;
	// }

	void Init(int model_index, int af_index, int lf_index)
	{
		this->af_index = af_index;
		this->lf_index = lf_index;
		this->model_index = model_index;
		model = models[model_index].model();
		lr = models[model_index].lr * activation_functions[af_index].lr_modifier;
		af = activation_functions[af_index].af();
		lf = loss_functions[lf_index].lf();
		model->Init(input_size, hidden_layer_size, output_size, af, min_weight_init, max_weight_init, min_clamp, max_clamp);
		last_predicted_stick_x.resize(history_size);
		last_predicted_stick_y.resize(history_size);
		last_stick_x.resize(history_size, 0.000001);
		last_stick_y.resize(history_size, 0.000001);
		rolling_window.resize(window_size);
	}

	void Randomize()
	{
		model->Randomize();
		if (af->IsTrainable())
		{
			af->Randomize();
		}
	}

	void Draw()
	{
		// anti-aliasing
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
		glLineWidth(10);
		float point_size = .025;
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);

		// draw the curve of the activation function
		glBegin(GL_LINE_STRIP);
		for (int i = 0; i < 1000; i++)
		{
			float x = map(i, 0, 1000, -1, 1);
			float y = af->Activate(x);
			glColor4f(0, 0, 1, 1);
			glVertex2f(x, y);
		}
		glEnd();

		// draw a fading line for the target
		glBegin(GL_LINE_STRIP);
		for (int i = 0; i < history_size; i++)
		{
			float alpha = map(i, 0, history_size, 0, 1);
			glColor4f(0, 1, 0, alpha);
			glVertex2f(last_stick_x[i], last_stick_y[i]);
		}
		glEnd();

		// render a single point for the target
		GraphicsGorilla::TwoD::Primitives::Filled::DrawCircle(point_size, 10, {last_stick_x[last_stick_x.size() - 1], last_stick_y[last_stick_y.size() - 1]}, {0, 1, 0, 1});

		// draw a fading line for the prediction
		glBegin(GL_LINE_STRIP);
		for (int i = 0; i < history_size; i++)
		{
			float alpha = map(i, 0, history_size, 0, 1);
			glColor4f(1, 0, 0, alpha);
			glVertex2f(last_predicted_stick_x[i], last_predicted_stick_y[i]);
		}
		glEnd();

		// render a single point for the prediction
		GraphicsGorilla::TwoD::Primitives::Filled::DrawCircle(point_size, 10, {last_predicted_stick_x[last_predicted_stick_x.size() - 1], last_predicted_stick_y[last_predicted_stick_y.size() - 1]}, {1, 0, 0, 1});

		float font_model_name_size = .5;
		float font_sub_name_size = .45;

		// draw the model name in the top left corner
		font.render(models[model_index].name, -1, .9, font_model_name_size, {.95, .95, .95, 1});
		// draw the activation function name under the model name
		font.render(activation_functions[af_index].name, -1, .8, font_sub_name_size, {.95, .95, .95, 1});
		// draw the loss function name under the activation function name
		font.render(loss_functions[lf_index].name, -1, .7, font_sub_name_size, {.95, .95, .95, 1});
		// draw the learning rate under the loss function name
		std::string lr_string = "LR: " + std::to_string(lr);
		font.render(lr_string, -1, .6, font_sub_name_size, {.95, .95, .95, 1});
		// draw the loss value under the loss function name
		std::string loss_string = "Loss: " + std::to_string(rolling_window[rolling_window.size() - 1]);
		font.render(loss_string, -1, .5, font_sub_name_size, {.95, .95, .95, 1});
		// draw the rolling average loss value under the loss value
		float loss_avg = 0;
		for (int i = 0; i < window_size; i++)
		{
			loss_avg += rolling_window[i];
		}
		loss_avg /= window_size;
		std::string loss_avg_string = "Loss Avg (" + std::to_string(window_size) + "): " + std::to_string(loss_avg);
		font.render(loss_avg_string, -1, .4, font_sub_name_size, {.95, .95, .95, 1});
		glDisable(GL_BLEND);
	}

	void Update(float stick_x, float stick_y)
	{
		std::vector<float> input = {(float)last_stick_x[0], (float)last_stick_y[0]};
		std::vector<float> target = {(float)stick_y, (float)stick_x};
		std::vector<float> output = model->Forward(input);
		last_stick_x.push_back(stick_x);
		last_stick_y.push_back(stick_y);
		if (last_stick_x.size() > history_size)
		{
			last_stick_x.erase(last_stick_x.begin());
			last_stick_y.erase(last_stick_y.begin());
		}
		last_predicted_stick_x.push_back(output[0]);
		last_predicted_stick_y.push_back(output[1]);
		if (last_predicted_stick_x.size() > history_size)
		{
			last_predicted_stick_x.erase(last_predicted_stick_x.begin());
			last_predicted_stick_y.erase(last_predicted_stick_y.begin());
		}
		float loss = lf->Calculate(output, target);
		// lr=(log(loss)+5)/(loss+10);
		rolling_window.push_back(loss);
		if (rolling_window.size() > window_size)
		{
			rolling_window.erase(rolling_window.begin());
		}
		float loss_avg = 0;
		for (int i = 0; i < window_size; i++)
		{
			loss_avg += rolling_window[i];
		}
		loss_avg /= window_size;
		Train(input, target, lr);
	}

	float getLearningRate()
	{
		return lr;
	}

	void setLearningRate(float lr)
	{
		this->lr = lr;
	}
};

float lerp(float a, float b, float t)
{
	return a + t * (b - a);
}

int main()
{
	srand(time(NULL));
	int width = 1920;
	int height = 1080;

	GraphicsGorilla::Window window(width, height, "MLManticore");

	if (!font.load("PlayfairDisplay-VariableFont_wght.ttf"))
	{
		printf("Failed to load font\n");
	}

	std::vector<Trainer> trainers;

	for (int model_index = 0; model_index < models.size(); model_index++)
	{
		for (int af_index = 0; af_index < activation_functions.size(); af_index++)
		{
			for (int lf_index = 0; lf_index < loss_functions.size(); lf_index++)
			{
				trainers.insert(trainers.begin(), Trainer());
				trainers.front().Init(model_index, af_index, lf_index);
			}
		}
	}
	printf("\n\n\n\nTrainer count:%ld\n", trainers.size());

	int selected_trainer = -1;

	window.SetDrawFunction([&](int screen_width, int screen_height)
						   {
							   // setup viewport
							   glViewport(0, 0, screen_width, screen_height);
							   // clear color buffer
							   glClear(GL_COLOR_BUFFER_BIT);

								if(selected_trainer == -1){
									// draw the training models in a grid
									// calculat the closest square root
									int rows = sqrt(trainers.size());
									int cols =std::ceil((float)trainers.size() / (float)rows);
									float width = 1.0 / (float)cols;
									float height = 1.0 / (float)rows;
									for (int i = 0; i < trainers.size(); i++)
									{
										int x = i % cols;
										int y = i / cols;
											//swap x so that the grid is drawn from left to right
											//x = cols - x - 1;
											//swap y so that the grid is drawn from top to bottom
											y = rows - y - 1;

										GraphicsGorilla::FrameBuffer fb;
										fb.Init(screen_width, screen_height);
										fb.DrawToFB({0.1, 0.1, 0.1, 1}, [&]()
													{ trainers[i].Draw(); });
										glColor3f(1, 1, 1);
										float x_pos = (float)x * width;
										float y_pos = (float)y * height;
										x_pos*=2;
										y_pos*=2;
										x_pos-=1;
										y_pos-=1;
										float x_size = width*2;
										float y_size = height*2;
										fb.DrawFBToScreen({x_pos, y_pos}, {x_size, y_size});
									} 
								}else{
									trainers[selected_trainer].Draw();
								} });

	static bool enable_controller = false;

	static long key_delay = 800;
	static long last_space_down = 0;
	static long last_left_down = 0;
	static long last_right_down = 0;
	static long last_up_down = 0;
	static bool last_down_down = 0;
	static float stick_x = 0;
	static float stick_y = 0;

	window.SetUpdateFunction([&](GLFWwindow *glwindow)
							 {
															 GLFWgamepadstate state;
		if (glfwGetGamepadState(GLFW_JOYSTICK_1, &state))
		{
			if(state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS)
			{
				enable_controller=true;
				stick_x = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
				stick_y = -state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
			}else{
				enable_controller=false;
			}
		}
		if(!enable_controller){
			stick_x = sin(glfwGetTime());
			stick_y = cos(glfwGetTime());
		}

		if(stick_x==0){
			stick_x = 0.0000001;
		}
		if(stick_y==0){
			stick_y = 0.0000001;
		}

		for (int i = 0; i < trainers.size(); i++)
		{
			trainers[i].Update(stick_x, stick_y);
		} });

	window.SetKeyCallback([&](GLFWwindow *glwindow, int key, int scancode, int action, int mods)
						  {
							  // check key presses using glfw
							  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
							  {
								  glfwSetWindowShouldClose(glwindow, GLFW_TRUE);
							  }

							  long current_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

							  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
							  {
								  if (glfwGetKey(glwindow, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
								  {
									  selected_trainer = -1;
								  }
								  else
								  {
									  selected_trainer++;
									  if (selected_trainer >= trainers.size())
									  {
										  selected_trainer = -1;
									  }
								  }
							  }

							  if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
							  {
								  selected_trainer--;
								  if (selected_trainer < -1)
								  {
									  selected_trainer = trainers.size() - 1;
								  }
							  }

							  if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
							  {
								  selected_trainer++;
								  if (selected_trainer >= trainers.size())
								  {
									  selected_trainer = -1;
								  }
							  }

							  if (key == GLFW_KEY_UP && action == GLFW_PRESS)
							  {
								  if (selected_trainer != -1)
								  {
									  if (glfwGetKey(glwindow, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS && glfwGetKey(glwindow, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
									  {
										  trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() + 0.0001);
									  }
									  else if (glfwGetKey(glwindow, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
									  {
										  trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() + 0.1);
									  }
									  else if (glfwGetKey(glwindow, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
									  {
										  trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() + 0.001);
									  }
									  else
									  {
										  trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() + 0.01);
									  }
								  }
							  }

							  if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
							  {
								  if (selected_trainer != -1)
								  {
									  if (glfwGetKey(glwindow, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS && glfwGetKey(glwindow, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
									  {
										  trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() - 0.0001);
									  }
									  else if (glfwGetKey(glwindow, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
									  {
										  trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() - 0.1);
									  }
									  else if (glfwGetKey(glwindow, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
									  {
										  trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() - 0.001);
									  }
									  else
									  {
										  trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() - 0.01);
									  }
								  }
							  }

							  if (key == GLFW_KEY_R && action == GLFW_PRESS)
							  {
								  if (selected_trainer != -1)
								  {
									  trainers[selected_trainer].Randomize();
								  }
							  } });

	window.SetControllerCallback([&](GLFWwindow *window, int jid, int event)
								 {
									if (event == GLFW_CONNECTED)
									{
										printf("Controller connected\n");
									}
									else if (event == GLFW_DISCONNECTED)
									{
										printf("Controller disconnected\n");
									} });

	window.SetMouseButtonCallback([&](GLFWwindow *window, int button, int action, int mods)
								  {
									 double xpos, ypos;
									 glfwGetCursorPos(window, &xpos, &ypos);
									 int width, height;
									 glfwGetWindowSize(window, &width, &height);
									 xpos = xpos / width * 2 - 1;
									 ypos = ypos / height * 2 - 1;
									 if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
									 {
										int rows = sqrt(trainers.size());
										int cols =std::ceil((float)trainers.size() / (float)rows);
										float width = 1.0 / (float)cols;
										float height = 1.0 / (float)rows;
										for (int i = 0; i < trainers.size(); i++)
										{
											int x = i % cols;
											int y = i / cols;
											float trainer_x_pos = (float)x * width;
											float trainer_y_pos = (float)y * height;
											trainer_x_pos*=2;
											trainer_y_pos*=2;
											trainer_x_pos-=1;
											trainer_y_pos-=1;
											float x_size = width*2;
											float y_size = height*2;
											//check if the mouse is inside the box
											if(xpos>trainer_x_pos && xpos<trainer_x_pos+x_size && ypos>trainer_y_pos && ypos<trainer_y_pos+y_size)
											{
												selected_trainer = i;
											}
										}
									 } });

	window.SetMouseScrollCallback([&](GLFWwindow *window, double xoffset, double yoffset)
								  {
		if (selected_trainer != -1)
		{
			if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS && glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			{
				if (yoffset > 0)
				{
					trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() * 1.1);
				}
				else
				{
					trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() * 0.9);
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
			{
				if (yoffset > 0)
				{
					trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() * 1.75);
				}
				else
				{
					trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() * 0.25);
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			{
				if (yoffset > 0)
				{
					trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() * 1.15);
				}
				else
				{
					trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() * 0.85);
				}
			}
			else
			{
				if (yoffset > 0)
				{
					trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() * 1.5);
				}
				else
				{
					trainers[selected_trainer].setLearningRate(trainers[selected_trainer].getLearningRate() * 0.5);
				}
			}
		} });

	window.Run();

	return 0;
}