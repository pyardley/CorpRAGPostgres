import React from 'react';
import { helper } from '../shared/utils';

export function run(value) {
  return helper(value) + (React ? 0 : 0);
}
