# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frequency analysis of image classification models.

Add frequency basis vectors as perturbations to the inputs of image
classification models. Analyze how the output changes across different frequency
components.
"""

import numpy as np
import cv2
import pdb

def _get_symmetric_pos(dims, pos):
  """Compute the symmetric position of the point in 2D FFT.

  Args:
    dims: a tuple of 2 positive integers, dimensions of the 2D array.
    pos: a tuple of 2 integers, coordinate of the query point.

  Returns:
    a numpy array of shape [2], the coordinate of the symmetric point of the
      query point.
  """
  x = np.array(dims)
  p = np.array(pos)
  return np.where(np.mod(x, 2) == 0, np.mod(x - p, x), x - 1 - p)


def _get_fourier_basis(x, y):
  """Compute real-valued basis vectors of 2D discrete Fourier transform.

  Args:
    x: first dimension of the 2D numpy array.
    y: second dimension of the 2D numpy array.

  Returns:
    fourier_basis, a real-valued numpy array of shape (x, y, x, y).
      Each 2D slice of the last two dimensions of fourier_basis,
      i.e., fourier_basis[i, j, :, :] is a 2D array with unit norm,
      such that its Fourier transform, when low frequency located
      at the center, is supported at (i, j) and its symmetric position.
  """
  # If Fourier basis at (i, j) is generated, set marker[i, j] = 1.
  marker = np.zeros([x, y], dtype=np.uint8)
  fourier_basis = np.zeros([x//20+1, y//20+1, x, y], dtype=np.float32)
  for i in range(0,x,20):
    for j in range(0,y,20):
      if marker[i, j] > 0:
        continue
      freq = np.zeros([x, y], dtype=np.complex64)
      sym = _get_symmetric_pos((x, y), (i, j))
      sym_i = sym[0]
      sym_j = sym[1]
      if (sym_i, sym_j) == (i, j):
        freq[i, j] = 1.0
        marker[i, j] = 1
      else:
        freq[i, j] = 0.5 + 0.5j
        freq[sym_i, sym_j] = 0.5 - 0.5j
        marker[i, j] = 1
        marker[sym_i, sym_j] = 1
      basis = np.fft.ifft2(np.fft.ifftshift(freq))
      basis = np.sqrt(x * y) * np.real(basis)

      fourier_basis[i//20, j//20, :, :] = basis
      if (sym_i, sym_j) != (i, j):
        fourier_basis[sym_i//20, sym_j//20, :, :] = basis
  return fourier_basis


def _get_sum_of_norms(a_list):
  """Compute sum of row norms of the reshaped 2D arrays in a_list.

  For each numpy array in a_list, this function first reshape the array to 2D
  with shape (batch_size, ?), then compute the l_2 norm of each row, i.e.,
  norm of [i, :] for i in range(batch_size), and then compute the sum of the row
  norms. This is equivalent to computing the l_(2,1) norm of the transpose of
  the 2D matrices.

  Args:
    a_list: a list of numpy arrays, each with shape (batch_size, ...)

  Returns:
    a list of sum of l_2 row norms.
  """
  sum_of_norms = []
  for each_array in a_list:
    array_2d = each_array.reshape((each_array.shape[0], -1))
    norms = np.linalg.norm(array_2d, axis=1)
    sum_of_norms.append(np.sum(norms))
  return sum_of_norms


def _generate_perturbed_images(images,
                               perturb_basis,
                               perturb_norm = 1.0,
                               clip_min = None,
                               clip_max = None,
                               rand_flip = False):
  """Generate perturbed images given a perturbation basis.

  Args:
    images: numpy array, clean images before perturbation. The shape of this
      array should be [batch_size, height, width] or
      [batch_size, height, width, num_channels].
    perturb_basis: numpy array basis matrix for perturbation. The shape of this
      array should be [height, width]. It is assumed to have unit l_2 norm.
    perturb_norm: the l_2 norm of the Fourier-basis perturbations.
    clip_min: lower bound for clipping operation after adding perturbation. If
      None, no lower clipping.
    clip_max: upper bound for clipping operation after adding perturbation. If
      None, no upper clipping.
    rand_flip: whether or not to randomly flip the sign of basis vectors.

  Returns:
    perturbed images, numpy array with the same shape as clean_images.
  """
  if len(images.shape) != 3 and len(images.shape) != 4:
    raise ValueError('Incorrect shape of clean images.')

  if len(images.shape) == 3:
    clean_images = np.expand_dims(images, axis=3)
  elif len(images.shape) == 4:
    clean_images = images
  else:
    raise ValueError('Unexpected number of dimensions: %d' % len(images.shape))

  batch_size = clean_images.shape[0]
  num_channels = clean_images.shape[3]

  if not rand_flip:
    # (batch, height, width, channel) -> (batch, channel, height, width)
    clean_images_t = np.transpose(clean_images, (0, 3, 1, 2))
    perturb_images_t = clean_images_t + perturb_norm * perturb_basis
    # (batch, channel, height, width) -> (batch, height, width, channel)
    perturb_images = np.transpose(perturb_images_t, (0, 2, 3, 1))
  else:
    # Add random flips when adding basis vectors.
    flip = 2.0 * np.random.binomial(
        1, 0.5, size=batch_size * num_channels) - 1.0
    flat_basis = np.reshape(perturb_basis, (-1))
    perturbation = np.reshape(np.outer(flip, flat_basis),
                              (batch_size, num_channels) + perturb_basis.shape)
    # (batch, channel, height, width) -> (batch, height, width, channel)
    perturbation = np.transpose(perturbation, (0, 2, 3, 1))
    perturb_images = clean_images + perturbation

  if clip_min is not None or clip_max is not None:
    perturb_images = np.clip(perturb_images, clip_min, clip_max)

  if len(images.shape) == 3:
    return np.squeeze(perturb_images, axis=3)
  else:
    return perturb_images


if __name__=="__main__":
  fourier_basis=_get_fourier_basis(224,224)
  np.save("fourier_basis_sample/fourier_basis224.npy",fourier_basis)
