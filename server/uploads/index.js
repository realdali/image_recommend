var fs = require("fs"),
    path = require("path");

function walk(dir, callback) {
    fs.readdir(dir, function(err, files) {
        if (err) throw err;
        files.forEach(function(file) {
            var filepath = path.join(dir, file);
            fs.stat(filepath, function(err,stats) {
                if (stats.isDirectory()) {
                    walk(filepath, callback);
                } else if (stats.isFile()) {
                    callback(filepath, stats);
                }
            });
        });
    });
}


walk('Shoes', function(filepath, stats) {
  const name = path.basename(filepath);
  console.log(filepath, name)
  fs.copyFileSync(filepath, './Shoes/' + name)
});


walk('Slippers', function(filepath, stats) {
  const name = path.basename(filepath);
  console.log(filepath, name)
  fs.copyFileSync(filepath, './Slippers/' + name)
});


walk('Boots', function(filepath, stats) {
  const name = path.basename(filepath);
  console.log(filepath, name)
  fs.copyFileSync(filepath, './Boots/' + name)
});


walk('Sandals', function(filepath, stats) {
  const name = path.basename(filepath);
  console.log(filepath, name)
  fs.copyFileSync(filepath, './Sandals/' + name)
});
