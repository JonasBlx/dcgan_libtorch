#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <filesystem>  // Pour vérifier l'existence des fichiers

// Définition du module du générateur
struct GeneratorImpl : torch::nn::Module {
    torch::nn::ConvTranspose2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    torch::nn::BatchNorm2d batch_norm1{nullptr}, batch_norm2{nullptr}, batch_norm3{nullptr};

    GeneratorImpl(int kNoiseSize) {
        conv1 = register_module("conv1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)));
        batch_norm1 = register_module("batch_norm1", torch::nn::BatchNorm2d(256));
        conv2 = register_module("conv2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)));
        batch_norm2 = register_module("batch_norm2", torch::nn::BatchNorm2d(128));
        conv3 = register_module("conv3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)));
        batch_norm3 = register_module("batch_norm3", torch::nn::BatchNorm2d(64));
        conv4 = register_module("conv4", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::tanh(conv4(x));
        return x;
    }
};
TORCH_MODULE(Generator);  // Wrapper pour permettre l'utilisation de shared_ptr pour GeneratorImpl

// Définition du module du discriminateur
struct DiscriminatorImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    torch::nn::BatchNorm2d batch_norm1{nullptr}, batch_norm2{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr};

    DiscriminatorImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)));
        batch_norm1 = register_module("batch_norm1", torch::nn::BatchNorm2d(128));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)));
        batch_norm2 = register_module("batch_norm2", torch::nn::BatchNorm2d(256));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)));
        leaky_relu = register_module("leaky_relu", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = leaky_relu(conv1(x));
        x = leaky_relu(batch_norm1(conv2(x)));
        x = leaky_relu(batch_norm2(conv3(x)));
        x = torch::sigmoid(conv4(x));
        return x;
    }
};
TORCH_MODULE(Discriminator);

void weights_init(torch::nn::Module& module) {
    if (auto* conv = module.as<torch::nn::Conv2d>()) {
        torch::nn::init::normal_(conv->weight, 0.0, 0.02);
    } else if (auto* bn = module.as<torch::nn::BatchNorm2d>()) {
        torch::nn::init::normal_(bn->weight, 1.0, 0.02);
        torch::nn::init::constant_(bn->bias, 0);
    }
}

int main() {
    torch::manual_seed(0);  // Pour la reproductibilité

    int kNoiseSize = 100;
    int kBatchSize = 64;
    int kNumberOfEpochs = 30;
    int kCheckpointEvery = 200;  // Intervalle de sauvegarde des checkpoints
    int checkpoint_counter = 0;

    Generator generator(kNoiseSize);
    Discriminator discriminator;

    generator->apply(weights_init);
    discriminator->apply(weights_init);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    generator->to(device);
    discriminator->to(device);

    torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(2e-3).betas(std::make_tuple(0.5, 0.5)));
    torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(2e-3).betas(std::make_tuple(0.5, 0.5)));

    // Vérifiez si les checkpoints existent et chargez-les si disponibles
    if (std::filesystem::exists("generator-checkpoint.pt") && std::filesystem::exists("discriminator-checkpoint.pt")) {
        std::cout << "Loading checkpoints..." << std::endl;
        torch::load(generator, "generator-checkpoint.pt");
        torch::load(discriminator, "discriminator-checkpoint.pt");
        torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
        torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
        std::cout << "Checkpoints loaded." << std::endl;
    }

    auto dataset = torch::data::datasets::MNIST("./data/MNIST/raw")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    std::cout << "Starting data loading..." << std::endl;
    auto data_loader = torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));
    std::cout << "Data loading completed." << std::endl;

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        std::cout << "Starting epoch " << epoch << " of " << kNumberOfEpochs << std::endl;
        int64_t batch_index = 0;
        for (torch::data::Example<>& batch : *data_loader) {

            std::cout << "Processing batch " << ++batch_index << std::endl;

            // Train discriminator with real images
            discriminator->zero_grad();
            auto real_images = batch.data.to(device);
            auto real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
            auto output_real = discriminator->forward(real_images).reshape(real_labels.sizes());
            auto loss_real = torch::binary_cross_entropy_with_logits(output_real, real_labels);
            loss_real.backward();

            // Train discriminator with fake images
            auto noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
            auto fake_images = generator->forward(noise);
            auto fake_labels = torch::zeros(batch.data.size(0), device);
            auto output_fake = discriminator->forward(fake_images.detach()).reshape(fake_labels.sizes());
            auto loss_fake = torch::binary_cross_entropy_with_logits(output_fake, fake_labels);
            loss_fake.backward();

            auto d_loss = loss_real + loss_fake;
            discriminator_optimizer.step();

            // Train generator
            generator->zero_grad();
            fake_labels.fill_(1);
            output_fake = discriminator->forward(fake_images).reshape(fake_labels.sizes());
            auto g_loss = torch::binary_cross_entropy_with_logits(output_fake, fake_labels);
            g_loss.backward();
            generator_optimizer.step();

            if (batch_index % kCheckpointEvery == 0) {
                std::cout << "Saving checkpoints at batch " << batch_index << " of epoch " << epoch << std::endl;
                torch::save(generator, "generator-checkpoint.pt");
                torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
                torch::save(discriminator, "discriminator-checkpoint.pt");
                torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
                // Sample the generator and save the images.
                torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
                torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
                std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
            }

            std::cout << "Epoch [" << epoch << "/" << kNumberOfEpochs << "] "
                      << "Batch [" << batch_index << "] "
                      << "D_loss: " << d_loss.item<float>() << " "
                      << "G_loss: " << g_loss.item<float>() << std::endl;
        }

        // Sauvegarde à chaque fin d'epoch
        std::cout << "Saving checkpoints for epoch " << epoch << std::endl;
        torch::save(generator, "generator_checkpoint.pt");
        torch::save(discriminator, "discriminator_checkpoint.pt");
        torch::save(generator_optimizer, "generator_optimizer_checkpoint.pt");
        torch::save(discriminator_optimizer, "discriminator_optimizer_checkpoint.pt");
        std::cout << "Checkpoints saved." << std::endl;
    }

    // Sauvegarde finale après la fin de l'entraînement
    std::cout << "Saving final models..." << std::endl;
    torch::save(generator, "generator_checkpoint.pt");
    torch::save(discriminator, "discriminator_checkpoint.pt");
    torch::save(generator_optimizer, "generator_optimizer_checkpoint.pt");
    torch::save(discriminator_optimizer, "discriminator_optimizer_checkpoint.pt");
    std::cout << "Final models saved." << std::endl;

    std::cout << "Training completed." << std::endl;

    return 0;
}
