import tensorflow as tf
from tensorflow import keras

class CycleGan(keras.Model):
    def __init__(
            self,
            monet_generator,
            photo_generator,
            monet_discriminator,
            photo_discriminator,
            lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

    def compile(
            self,
            m_gen_optimizer,
            p_gen_optimizer,
            m_disc_optimizer,
            p_disc_optimizer,
            gen_loss_fn,
            disc_loss_fn,
            cycle_loss_fn,
            identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        self.loss_trackers = dict(
            monet_gen_loss=keras.metrics.Mean(name="monet_gen_loss"),
            monet_disc_loss=keras.metrics.Mean(name="monet_disc_loss"),
            photo_gen_loss=keras.metrics.Mean(name="photo_gen_loss"),
            photo_disc_loss=keras.metrics.Mean(name="photo_disc_loss"),
            total_loss=keras.metrics.Mean(name="total_loss"),
                                  )

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_trackers[key] for key in sorted(self.loss_trackers.keys())]

    @tf.function
    def train_step(self, batch_data):
        real_monet, real_photo = batch_data

        if self.compiled_loss is not None:
            ValueError("You passed a loss function to `model.compile` however we are defining our own losses inside the"
                       "training loop. Passed loss: {}".format(self.compiled_loss))
        if self.compiled_metrics is not None:
            ValueError("You passed metrics to the compile function, however we are defining our own metrics in the "
                       "training loop. Passed metrics: {}".format(self.compiled_metrics))


        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(
                real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet,
                                                                                             self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo,
                                                                                             self.lambda_cycle)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

            both_disc_loss = monet_disc_loss + photo_disc_loss
            signal_disc_improving = both_disc_loss < self.loss_trackers['monet_disc_loss'].result() + self.loss_trackers['photo_disc_loss'].result()
            monet_disc_loss = tf.where(signal_disc_improving, 0., monet_disc_loss)
            photo_disc_loss = tf.where(signal_disc_improving, 0., photo_disc_loss)


        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))
        #These keep track of the mean loss over the epoch. THey are reset in the model.fit function at the start of each
        # epoch automatically.
        self.loss_trackers["monet_gen_loss"].update_state(total_monet_gen_loss)
        self.loss_trackers["photo_gen_loss"].update_state(total_photo_gen_loss)
        self.loss_trackers["monet_disc_loss"].update_state(monet_disc_loss)
        self.loss_trackers["photo_disc_loss"].update_state(photo_disc_loss)
        self.loss_trackers["total_loss"].update_state(total_monet_gen_loss+total_photo_gen_loss+monet_disc_loss+photo_disc_loss)
        return {k:v.result() for k,v in self.loss_trackers.items()}

    @tf.function
    def test_step(self, batch_data):
        #similar to the train_step except for skipping the optimisation
        real_monet, real_photo = batch_data

        if self.compiled_loss is not None:
            ValueError("You passed a loss function to `model.compile` however we are defining our own losses inside the"
                       "training loop. Passed loss: {}".format(self.compiled_loss))
        if self.compiled_metrics is not None:
            ValueError("You passed metrics to the compile function, however we are defining our own metrics in the "
                       "training loop. Passed metrics: {}".format(self.compiled_metrics))

        # photo to monet back to photo
        fake_monet = self.m_gen(real_photo, training=True)
        cycled_photo = self.p_gen(fake_monet, training=True)

        # monet to photo back to monet
        fake_photo = self.p_gen(real_monet, training=True)
        cycled_monet = self.m_gen(fake_photo, training=True)

        # generating itself
        same_monet = self.m_gen(real_monet, training=True)
        same_photo = self.p_gen(real_photo, training=True)

        # discriminator used to check, inputing real images
        disc_real_monet = self.m_disc(real_monet, training=True)
        disc_real_photo = self.p_disc(real_photo, training=True)

        # discriminator used to check, inputing fake images
        disc_fake_monet = self.m_disc(fake_monet, training=True)
        disc_fake_photo = self.p_disc(fake_photo, training=True)

        # evaluates generator loss
        monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
        photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

        # evaluates total cycle consistency loss
        total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(
            real_photo, cycled_photo, self.lambda_cycle)

        # evaluates total generator loss
        total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet,
                                                                                         self.lambda_cycle)
        total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo,
                                                                                         self.lambda_cycle)

        # evaluates discriminator loss
        monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
        photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # These keep track of the mean loss over the epoch. THey are reset in the model.fit function at the start of each
        # epoch automatically.
        self.loss_trackers["monet_gen_loss"].update_state(total_monet_gen_loss)
        self.loss_trackers["photo_gen_loss"].update_state(total_photo_gen_loss)
        self.loss_trackers["monet_disc_loss"].update_state(monet_disc_loss)
        self.loss_trackers["photo_disc_loss"].update_state(photo_disc_loss)
        self.loss_trackers["total_loss"].update_state(
            total_monet_gen_loss + total_photo_gen_loss + monet_disc_loss + photo_disc_loss)
        return {k: v.result() for k, v in self.loss_trackers.items()}


def build_discriminator_loss(strategy):
    with strategy.scope():
        def discriminator_loss(real, generated):
            real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
                tf.ones_like(real), real)

            generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)(
                tf.zeros_like(generated), generated)

            total_disc_loss = real_loss + generated_loss

            return total_disc_loss * 0.5
    return discriminator_loss


def build_generator_loss(strategy):
    with strategy.scope():
        def generator_loss(generated):
            return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
                tf.ones_like(generated), generated)
    return generator_loss

def build_calc_cycle_loss(strategy):
    with strategy.scope():
        def calc_cycle_loss(real_image, cycled_image, LAMBDA):
            loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
            return LAMBDA * loss1
    return  calc_cycle_loss

def build_identity_loss(strategy):
    with strategy.scope():
        def identity_loss(real_image, same_image, LAMBDA):
            loss = tf.reduce_mean(tf.abs(real_image - same_image))
            return LAMBDA * 0.5 * loss
    return identity_loss