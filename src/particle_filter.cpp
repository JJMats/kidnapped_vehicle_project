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

std::default_random_engine gen;

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
  
  /*
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  */
  
  for(int i=0; i < num_particles; ++i){
    Particle p = {
      id: i,
      x: dist_x(gen),
      y: dist_y(gen),
      theta: dist_theta(gen),
      weight: 1.0      
    };
    
    this.particles.push_back(p);
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
  if(fabs(yaw_rate) < 0.00001){
    if(yaw_rate > 0.0){
      yaw_rate = 0.00001;
    } else {
      yaw_rate = -0.00001;
    }
  }
  double turn_radius = velocity / yaw_rate;
  
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  // I can also iterate through each particle with num_particles. Which method has less overhead?
  for (auto particle : particles){
    //double new_x = update_x(particle.x, turn_radius, particle.theta, yaw_rate, delta_t);
    particle.x += particle.x + turn_radius * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
    //double new_y = update_y(particle.y, turn_radius, particle.theta, yaw_rate, delta_t);
    particle.y = particle.y + turn_radius * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
    //double new_theta = update_theta(particle.theta, yaw_rate, delta_t);
    particle.theta = particle.theta + yaw_rate * delta_t;
      
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
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
  /*
  double min_distance = std::numeric_limits<double>::max();
  int closest_obs_index = -1;
  for(int i=0; i < observations.size(); ++i){
    float dist = dist(observations[i].x, observations[i].y,
                      predicted.x, predicted.y);
    if(dist < min_distance)
      closest_obs_index = i;
      min_distance = dist;
  }
  std::cout << "Closest observation:" << observations[closest_obs_index] << "; Distance: " << min_distance << std::endl;
  //return observations[closest_obs_index];
  */
  
  // Loop through observations to find the closest predicted particle for each observation
  for(int i=0; i < observations.size(); ++i){
    double min_distance = std::numeric_limits<double>::max();
    int closest_particle_id = -1;
    
    // Loop through predicted LandmarkObs vector to determine the closest particle distance to the observation
    for(int j=0; j < predicted.size(); ++j){
      double obs_distance = dist(observations[i].x, observations[i].y,
                         predicted.x, predicted.y);
      
      // Store the minimum distance and particle id if the distance is less than min_distance
      if(obs_distance < min_distance){
        min_distance = obs_distance;
        closest_particle_id = predicted.id;
      }
    }
    
    // Set the observation's id to the closest particle id
    observations[i].id = closest_particle_id;
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
  
  for(auto particle : particles){
    // Instantiate a vector to store landmarks that are within the sensor_range distance of the vehicle
    vector<LandmarkObs> predictions;
      
    // Loop through landmarks to determine if they are within the sensor_range distance of the vehicle
    for(auto landmark : map_landmarks.landmark_list){
      p_to_lm_dist = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
      if(p_to_lm_dist <= sensor_range){
        LandmarkObs lm = {
          id: landmark.id_i,
          x: landmark.x_f,
          y: landmark.y_f
        };
        predictions.push_back(lm);
      }
    }
    
    // Calculate sin and cos of the particle's heading prior to the transformation loop
    double sin_theta = sin(particle.theta);
    double cos_theta = cos(particle.theta);
    
    // Instantiate a vector to store observations converted from vehicle to map coordinates via homogenous transformation
    vector<LandmarkObs> ht_observations;
    for(auto observation : observations){
      double new_x = particle.x + (cos_theta * observation.x) - (sin_theta * observation.y);
      double new_y = particle.y + (sin_theta * observation.x) + (cos_theta * observation.y);
      LandmarkObs transformed_obs = {
        id: observation.id,
        x: new_x,
        y: new_y
      };
      ht_observations.push_back(transformed_obs);
    }
    
    // Pass the selected predictions and transformed observations to the dataAssociation function to determine the closest particle id
    dataAssociation(predictions, ht_observations);
    
    // Reinitialize the weight of the particle to 1.0
    particle.weight = 1.0;
    
    // Loop through transformed observations to calculate new particle weight via Multivariate-Gaussian Probability Density    
    for(auto ht_obs : ht_observations){
      for(auto prediction : predictions){
        if(prediction.id == ht_obs.id){
          double new_weight = normalizer * exp(-1.0 * ((pow(prediction.x - ht_obs.x, 2.0) / variance_x) + (pow(prediction.y - ht_obs.y, 2.0) / variance_y)));
          particle.weight *= new_weight;
          break;
        }
      }
    }
  }
  
  
  
  /*
  //for(int i=0; i < particles.size(); ++i){
  for(auto particle : particles){
    double final_weight = 1.0;
    double sin_theta = sin(particle.theta);
    double cos_theta = cos(particle.theta);
    
    for(int i=0; i < observations.size(); ++i){
      // Convert observations to map coordinates via Homogenous Transformation
      LandmarkObs predicted;
      double predicted.x = particle.x + (cos_theta * observation.x) - (sin_theta * observation.y);
      double predicted.y = particle.y + (sin_theta * observation.x) + (cos_theta * observation.y);
      
      // Get the closest landmark
      // Should I use the sensor_range variable here to determine whether or not the closest landmark coordinate is within this distance?
      vector<LandmarkObs> closest_landmark = dataAssociation(predicted, observations);
            
      // *** Create a vector of landmarks that are within sensor_range for each particle
      vector<LandmarkObs> landmarks_in_range;
      for(int m=0; m < map_landmarks.size(); ++m){
        if(dist(predicted.x, predicted.y, map_landmarks[m].x_f, map_landmarks[m].y_f) < sensor_range)
        landmarks_in_range.push_back();
      }
      
      // *** Send this vector into the dataAssociation function to associate the sensor measurements to map landmarks
      // *** Use the resulting associations to calculate the new weight of each particle uwing the MVGPD function
      
      
      // Calculate the Multivariate-Gaussian Probability Density for the observation and closest landmark
      // Append the MVGPD to the final_weight
      float weight = normalizer * exp(-1.0 * ((pow(predicted.x - closest_landmark.x, 2.0) / variance_x) + (pow(predicted.y - closest_landmark.y, 2.0) / variance_y)));
      final_weight *= weight;
    }
    // Set final weight of the particle
    particle.weight = final_weight;
  }
  // Associate each measurement with a landmark identifier
  //  - Take the closest landmark to each transformed observation
  
  // Calculate the weight value of the particle
  */
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
  
  // Set beta
  double beta = 0.0;
  
  // Get the maximum weight of the particle_weights vector
  double maximum_weight = max_element(particle_weights.begin(), particle_weights.end());
  
  // Randomly select an index to start from on the resampling wheel
  int index = rand() % num_particles;
  
  // Loop through resampling wheel to generate a vector of new particles
  for(int i=0; i < num_particles; ++i){
    beta += double(rand() % num_particles) * 2.0 * max_weight;
    while (beta > weights[index]){
      beta -= weights[index];
      index = (1 + index) % num_particles;      
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