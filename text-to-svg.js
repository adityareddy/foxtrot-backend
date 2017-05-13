#!/usr/bin/env node

/**
 * Copyright (c) 2016 Hideki Shiro
 */

const program = require('commander')
  .usage('[options] text')
  .option('-x, --x <n>', 'horizontal offset', parseFloat)
  .option('-y, --y <n>', 'vertical offset', parseFloat)
  .option('-s, --font-size <n>', 'font size', parseFloat)
  .option('-k, --kerning', 'kerning')
  .option('-a, --anchor [value]', 'anchor point')
  .parse(process.argv);

if(program.args.length < 1) {
  program.outputHelp();
  process.exit();
}

const text = program.args[0];
const options = {
  x: program.x || 0,
  y: program.y || 0,
  fontSize: program.size || 72,
  kerning: program.kerning,
  anchor: program.anchor || ''
};

const TextToSVG = require('text-to-svg');
const textToSVG = TextToSVG.loadSync();

console.log(textToSVG.getD(text, options));