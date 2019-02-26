/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // Create particles from normal distributions, with weights initialized to 1.0
  for(int i=0; i < num_particles; ++i){    
    Particle p = {
      id: i,
      x: dist_x(gen),
      y: dist_y(gen),
      theta: dist_theta(gen),
      weight: 1.0      
    };

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // Do not allow yaw_rates less than 0.00001
  if(fabs(yaw_rate) < 0.00001){
    if(yaw_rate > 0.0){
      yaw_rate = 0.00001;
    } else {
      yaw_rate = -0.00001;
    }
  }
  
  // Pre-calculate turn_radius for state calculation
  double turn_radius = velocity / yaw_rate;
  
  // Calculate new state for each particle  
  for (unsigned int i=0; i < particles.size(); ++i){
    double new_x = particles[i].x + turn_radius * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
    double new_y = particles[i].y + turn_radius * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
    double new_theta = particles[i].theta + yaw_rate * delta_t;
    
    normal_distribution<double> dist_x(new_x, std_pos[0]);
    normal_distribution<double> dist_y(new_y, std_pos[1]);
    normal_distribution<double> dist_theta(new_theta, std_pos[2]);
        
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  // Loop through observations to find the closest landmark for each observation
  for(unsigned int i=0; i < observations.size(); ++i){
    double min_distance = std::numeric_limits<double>::max();
    int closest_landmark_id = -1;
    
    // Loop through predicted LandmarkObs vector to determine the closest particle distance to the observation
    for(unsigned int j=0; j < predicted.size(); ++j){
      double obs_distance = dist(observations[i].x, observations[i].y,
                         predicted[j].x, predicted[j].y);
      
      // Store the minimum distance and particle id if the distance is less than min_distance
      if(obs_distance < min_distance){
        min_distance = obs_distance;
        closest_landmark_id = predicted[j].id;
      }
    }
    
    // Set the observation's id to the closest particle id
    observations[i].id = closest_landmark_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  // Pre-calculate normalizer and denominators for Multivariate-Gaussian Probability Density function
  double normalizer = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
  double variance_x = 2.0 * pow(std_landmark[0], 2.0);
  double variance_y = 2.0 * pow(std_landmark[1], 2.0);
  
  for(unsigned int i=0; i < particles.size(); ++i){
    // Instantiate a vector to store landmarks that are within the sensor_range distance of the vehicle
    vector<LandmarkObs> predictions;
      
    // Loop through landmarks to determine if they are within the sensor_range distance of the vehicle
    //   If they are, append them to the predictions vector
    for(unsigned int j=0; j < map_landmarks.landmark_list.size(); ++j){
      double p_to_lm_dist = dist(map_landmarks.landmark_list[j].x_f, 
                                 map_landmarks.landmark_list[j].y_f, 
                                 particles[i].x, particles[i].y);
      if(p_to_lm_dist <= sensor_range){
        LandmarkObs lm = {
          id: map_landmarks.landmark_list[j].id_i,
          x: map_landmarks.landmark_list[j].x_f,
          y: map_landmarks.landmark_list[j].y_f
        };
        predictions.push_back(lm);
      }
    }
    
    // Calculate sin and cos of the particle's heading prior to the transformation loop
    double sin_theta = sin(particles[i].theta);
    double cos_theta = cos(particles[i].theta);
    
    // Instantiate a vector to store observations converted from vehicle to map coordinates via homogenous transformation
    vector<LandmarkObs> ht_observations;
    for(unsigned int j=0; j < observations.size(); ++j){
      double new_x = particles[i].x + (cos_theta * observations[j].x) - (sin_theta * observations[j].y);
      double new_y = particles[i].y + (sin_theta * observations[j].x) + (cos_theta * observations[j].y);
      LandmarkObs transformed_obs = {
        id: observations[j].id,
        x: new_x,
        y: new_y
      };
      ht_observations.push_back(transformed_obs);
    }
    
    // Pass the selected predictions and transformed observations to the dataAssociation function to determine the closest particle id
    dataAssociation(predictions, ht_observations);
    
    // Reinitialize the weight of the particle to 1.0
    particles[i].weight = 1.0;
    
    // Loop through transformed observations to calculate new particle weight via Multivariate-Gaussian Probability Density    
    for(unsigned int j=0; j < ht_observations.size(); ++j){
      for(unsigned int k=0; k < predictions.size(); ++k){
        if(predictions[k].id == ht_observations[j].id){
          double new_weight = normalizer * exp(-1.0 * ((pow(predictions[k].x - ht_observations[j].x, 2.0) / variance_x) 
                                                       + (pow(predictions[k].y - ht_observations[j].y, 2.0) / variance_y)));
          particles[i].weight *= new_weight;
          break;
        }
      }
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // Update the particles to the Baysian posterior distribution
  // Implement Resampling Wheel
  vector<Particle> new_particles;    
  vector<double> particle_weights;
  for(int i=0; i < num_particles; ++i){
    particle_weights.push_back(particles[i].weight);
  }  

  // Get the maximum weight of the particle_weights vector
  double maximum_weight = *max_element(particle_weights.begin(), particle_weights.end());
  
  // Instantiate a uniform real distribution to sample weights from
  std::uniform_real_distribution<double> ur_dist(0.0, maximum_weight * 2.0);  
  
  // Randomly select an index to start from on the resampling wheel
  int index = rand() % num_particles;
  
  // Loop through resampling wheel to generate a vector of new particles
  double beta = 0.0;
  for(int i=0; i < num_particles; ++i){
    beta += ur_dist(gen);
    while (beta > particle_weights[index]){
      beta -= particle_weights[index];
      index = (index + 1) % num_particles;      
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}