# _plugins/group_data_injector.rb

Jekyll::Hooks.register :site, :post_read do |site|
  groups_data = site.data['homepage_groups']
  
  # Create a mapping from a simple ID to the group data
  group_map = {
    "大语言模型学习" => {"id" => "llm"},
    "生活杂记" => {"id" => "life"}
    # Add other mappings here if you create new groups
  }

  if groups_data
    site.collections['groups'].docs.each do |doc|
      # Find the corresponding group data for the current document
      group_name_to_find = group_map.find { |name, data| data["id"] == doc.basename_without_ext }
      
      if group_name_to_find
        group_data = groups_data.find { |g| g['group_name'] == group_name_to_find[0] }

        if group_data
          # Inject data into the document's front matter
          doc.data['title'] = group_data['group_name']
          doc.data['description'] = group_data['group_description']
          doc.data['header-img'] = group_data['group_image']
          doc.data['posts'] = group_data['posts']
        end
      end
    end
  end
end
