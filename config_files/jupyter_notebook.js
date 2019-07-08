IPython.keyboard_manager.command_shortcuts.add_shortcut('ctrl-b', function (event) {
        help : 'run all cells'
	IPython.notebook.execute_all_cells();
        return false;
});


IPython.keyboard_manager.command_shortcuts.add_shortcut('ctrl-shift-b', function (event) {
	help : 'restart kernel and run all cells'
      	IPython.notebook.kernel.restart();
	setTimeout(function(){ IPython.notebook.execute_all_cells(); }, 1000);
	return false;
});